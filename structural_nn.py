#!/bin/python

import os

import tqdm
import torch
import torch.nn.functional as F
import torchvision
import pandas as pd
import numpy as np

from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from efficientnet_pytorch import EfficientNet
from sklearn.impute import KNNImputer
from sklearn.preprocessing import power_transform, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from config import Config
from stratifiedgroupkf import StratifiedGroupKFold

torch.manual_seed(Config.random_state)
torch.cuda.manual_seed(Config.random_state)
np.random.seed(Config.random_state)
torch.backends.cudnn.benchmark = True


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, df):
        # Create one iterable that get be __getitemed__
        self.image_dir = image_dir
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        identifier = self.df.iloc[index]['IDENTIFIER']
        clinical_tensor = torch.tensor(
            self.df.iloc[index][['AGE', 'GENDER', 'VENTRATE', 'ATRIALRATE', 'PR', 'QTC']],
            dtype=torch.float32)
        oc = torch.tensor(
            self.df.iloc[index][['OUTCOME']],  # The OUTCOME column is hardcoded
            dtype=torch.float32)

        # Goes to image_tensor below
        image_path = os.path.join(self.image_dir, identifier + '.png')
        image = Image.open(image_path)

        # Transforms
        # Efficientnet B4: 380
        # Efficientnet B5: 456
        if Config.resize:
            resize = torchvision.transforms.Resize(380)
            image = resize(image)

        # Grayscale
        gscale = torchvision.transforms.Grayscale(num_output_channels=1)
        image = gscale(image)

        # to_tensor supports PIL images and numpy arrays
        image_tensor = torchvision.transforms.functional.to_tensor(image)
        normalize = torchvision.transforms.Normalize(
            [0.5], [0.5])
            # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet only
        image_tensor = normalize(image_tensor)

        return identifier, clinical_tensor, image_tensor, oc


class ClinicalMLP(torch.nn.Module):
    def __init__(self, n_clinical_vars):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_clinical_vars, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        return out


class CombinerMLP(torch.nn.Module):
    def __init__(self, feature_modalities):
        super().__init__()
        print('Total feature modalities:', feature_modalities)
        self.fc1 = torch.nn.Linear(64 + 32, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 1)

    def forward(self, clinical_tensor, image_tensor):
        x = torch.cat((clinical_tensor, image_tensor), dim=1)

        out = F.relu(self.fc1(x))
        if Config.dropout_rate:
            out = torch.nn.Dropout(Config.dropout_rate)(out)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out


class Combonet(torch.nn.Module):
    def __init__(self, n_clinical_vars, img_model_type):
        super().__init__()
        # Init the Clinical MLP
        self.fc_clinical = ClinicalMLP(n_clinical_vars)

        # Init the image model
        if img_model_type == 'densenet201':
            print('Image model: Densenet201')
            self.image_model = torchvision.models.densenet201(pretrained=True)
            self.image_model.features.conv0 = torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.image_model.classifier = torch.nn.Linear(1920, 64, bias=False)

        # This should take care of both classes and channels
        elif img_model_type == 'efficientnet':
            print('Image model: Efficientnet B4')
            self.image_model = EfficientNet.from_pretrained(
                'efficientnet-b4', in_channels=1, num_classes=64)

        elif img_model_type == 'resnet':
            print('Image model: Resnet50')
            self.image_model = torchvision.models.resnet50(pretrained=True)
            self.image_model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.image_model.fc = torch.nn.Linear(2048, 64, bias=False)

        else:
            print('Model name not recognized')
            raise NotImplementedError

        # Init the combiner model
        self.combiner = CombinerMLP(2)

    def forward(self, clinical_data, image_data):
        clinical_out = self.fc_clinical(clinical_data)
        image_out = self.image_model(image_data)
        combined_out = self.combiner(clinical_out, image_out)

        return combined_out


class StructuralClassifier:
    def __init__(self, img_model_type, outcome, epoch_offset=0, saved_model_path=None, explain=False):
        self.img_model_type = img_model_type

        print(f'Looking at {outcome}')
        self.outcome = outcome

        self.ecg_plot_dir = Config.dir_ecg_plots
        ecg_metrics = Config.file_ecg_metrics
        processed_metrics = (
            Config.file_ecg_metrics.replace('.pickle', '') +
            f'_{self.outcome}.pickle')

        self.all_outcomes = [
            'AORTIC_STENOSIS',
            'MITRAL_REGURGITATION']

        if not os.path.exists(processed_metrics):
            print('Sit back and relax')

            self.df_patients = pd.read_pickle(ecg_metrics)
            available_images = set([i.split('.')[0] for i in os.listdir(self.ecg_plot_dir)])
            self.df_patients = self.df_patients.query('IDENTIFIER in @available_images')

            # Reduce to a non-obscene VentRate
            self.df_patients = self.df_patients.query('VENTRATE <= 140')

            # Also rearranges columns
            self.format_outcome()
            self.preprocess_tabular_data()

            # Save dataframe
            print('Saving imputed / scaled dataframe')
            self.df_patients.to_pickle(processed_metrics, protocol=4)
        else:
            print(f'Using processed dataframe: {os.path.basename(processed_metrics)}')
            self.df_patients = pd.read_pickle(processed_metrics)

        # Just in case
        self.df_patients = self.df_patients.dropna()

        print('Batch size:', Config.batch_size)

        # For debugging
        # self.df_patients = self.df_patients.sample(1200)

        # Continue training checkpointed model and explainability
        self.saved_model_path = saved_model_path
        self.epoch_offset = epoch_offset  # For saving results
        self.explain = explain

        self.device = torch.device('cuda')

        # Create directories
        os.makedirs(f'Models/{self.outcome}', exist_ok=True)
        os.makedirs(f'OutputProbabilities/{self.outcome}', exist_ok=True)

    def image_dataloader_generator(self, df):
        image_dataset = CustomImageDataset(self.ecg_plot_dir, df)
        image_dataloader = torch.utils.data.DataLoader(
            image_dataset, batch_size=Config.batch_size,
            shuffle=True, num_workers=os.cpu_count() - 1, pin_memory=True)

        return image_dataloader

    def preprocess_tabular_data(self):
        self.df_patients = self.df_patients.drop(
            ['FILENAME', 'SAMPLERATE'], axis=1)
        self.df_patients = self.df_patients.set_index(
            ['MRN', 'ECHODate', 'SITENAME', 'IDENTIFIER']).sort_index()

        df = self.df_patients.copy()
        df = df.drop('TIME_DELTA', axis=1)

        # KNN imputation for a dataset this size takes twice the age
        # of the universe. I'm typing this after the big crunch
        if os.path.exists(f'Data/{self.outcome}_imputed.pickle'):
            print('Using pre imputed dataframe')
            df = pd.read_pickle(f'Data/{self.outcome}_imputed.pickle')
        else:
            print('Starting imputation')
            df = self.df_patients.copy()
            df = df.drop('TIME_DELTA', axis=1)
            imputed = KNNImputer().fit_transform(df)
            df = pd.DataFrame(imputed, index=df.index, columns=df.columns)
            df.to_pickle(f'Data/{self.outcome}_imputed.pickle')

        print('Scaling')
        # Leave the gender out of the scaling
        drop_cols = ['GENDER', 'OUTCOME']
        df = df.drop(drop_cols, axis=1)
        scaled = power_transform(StandardScaler().fit_transform(df))
        df = pd.DataFrame(scaled, index=df.index, columns=df.columns)

        # Restore to usability
        # NOTE The reset_index is crucial
        drop_cols.append('TIME_DELTA')
        df[drop_cols] = self.df_patients[drop_cols]
        self.df_patients = df.reset_index()

    def format_outcome(self):
        # Disgusting HACK for valvular disease
        # Eliminates need for a massive switch statement
        replacement_dict = {
            'Normal': 0,
            'Borderline': 0,
            'Moderate': 1,
            'Severe': 1
        }

        # Copy the outcome column and drop everything else
        self.df_patients = self.df_patients.dropna(subset=[self.outcome])
        s_outcome = self.df_patients[self.outcome].replace(replacement_dict)

        all_outcomes = ['AORTIC_STENOSIS', 'MITRAL_REGURGITATION']
        self.df_patients = self.df_patients.drop(all_outcomes, axis=1)
        self.df_patients = self.df_patients.assign(OUTCOME=s_outcome)

    def eval_model(self, model, dataloader, test_size, mode, epoch=None, fold=99):
        all_ids = []
        all_oc = []
        all_pred = []
        all_prob = []

        epoch_loss = 0
        criterion = torch.nn.BCEWithLogitsLoss()

        model.eval()
        for batch in tqdm.tqdm(dataloader):

            with torch.no_grad():

                with autocast():
                    identifiers = batch[0]
                    clinical = batch[1].cuda()
                    waves = batch[2].cuda()
                    oc = batch[3].cuda()

                    pred = model(clinical, waves)
                    loss = criterion(pred.squeeze(), oc.squeeze())

                    epoch_loss += loss.item() * pred.shape[0]

            all_ids.extend(identifiers)
            all_oc.extend(oc.cpu().numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())
            all_prob.extend(torch.sigmoid(pred).cpu().numpy())

        # Testing loss
        eval_loss = epoch_loss / test_size

        # AUROC
        y = [i[0] for i in all_oc]
        y_pred = [i.item() for i in all_prob]

        df_y = pd.DataFrame([y, y_pred]).T.dropna()
        df_y.index = pd.Series(all_ids)
        try:
            metric = roc_auc_score(df_y[0], df_y[1])
        except ValueError:
            print('Metric generation error')
            metric = 0
            breakpoint()

        # Save output probabilities
        if epoch is not None:
            mt_out = round(metric, 3)
            df_y.to_pickle(
                f'OutputProbabilities/{self.outcome}/C_{mode}_F{fold}_E{epoch}.pickle')

        return eval_loss, metric

    def get_splits(self, df):
        print('Stratified Mothereffin\' Group K Fold Splitting')
        gskf = StratifiedGroupKFold(
            n_splits=Config.cross_val_folds,
            shuffle=True,
            random_state=Config.random_state)
        splitter = gskf.split(
            df.drop('OUTCOME', axis=1),
            df['OUTCOME'],
            groups=df['MRN'])

        train_test_indices = []
        for train_idx, test_idx in splitter:
            train_test_indices.append((train_idx, test_idx))

        return train_test_indices

    def gaping_maw(self, dict_dataframes, fold):
        print('Starting fold', fold)

        df_train = dict_dataframes['train']
        df_test = dict_dataframes['test']
        df_int_val = dict_dataframes['int_val']
        df_ext_val = dict_dataframes['ext_val']

        # Track performance - Config.patience epochs must be considered before
        # stopping training and moving to the next fold
        performance_track = []
        extreme_metric = 0

        # Model
        # Going about loading models the long way since the usual way simply
        # does NOT seem to want to work
        if self.saved_model_path is not None:
            print('Loading saved model for continued training')

            model_l = torch.load(self.saved_model_path, map_location='cpu')
            state_dict = model_l.state_dict()
            prefix = 'module.'
            n_clip = len(prefix)
            adapted_dict = {
                k[n_clip:]: v for k, v in state_dict.items()
                if k.startswith(prefix)}

            model = Combonet(6, self.img_model_type)
            model.load_state_dict(adapted_dict)

        else:
            model = Combonet(6, self.img_model_type)

        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model.cuda()

        # Possible use of pos_weight parameter - skipping for now
        df_vc = df_train['OUTCOME'].value_counts(normalize=True)
        pos_weight = df_vc.loc[0] / df_vc.loc[1]
        print('pos_weight:', pos_weight)
        criterion = torch.nn.BCEWithLogitsLoss()
            # pos_weight=torch.tensor(pos_weight, dtype=torch.float32))

        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        # optim = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max', verbose=True)

        scaler = GradScaler()

        train_dataloader = self.image_dataloader_generator(df_train)
        test_dataloader = self.image_dataloader_generator(df_test)
        int_val_dataloader = self.image_dataloader_generator(df_int_val)
        ext_val_dataloader = self.image_dataloader_generator(df_ext_val)

        for epoch in range(Config.epochs):
            epoch_loss = 0

            model.train()
            for batch in tqdm.tqdm(train_dataloader):

                # Same as optim.zero_grad()
                for param in model.parameters():
                    param.grad = None

                with autocast():
                    # Identifier is not accessed here
                    clinical_batch = batch[1].cuda()
                    image_batch = batch[2].cuda()
                    oc_batch = batch[3].cuda()

                    pred = model(clinical_batch, image_batch)
                    loss = criterion(pred.squeeze(), oc_batch.squeeze())

                    epoch_loss += loss.item() * pred.shape[0]

                # Gradient scaling for AMP
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

            # Overall epoch loss
            training_loss = epoch_loss / df_train.shape[0]

            # Get progress every epoch
            print(f'Evaluating @ Epoch {epoch}')

            validation_loss, validation_metric = self.eval_model(
                model, int_val_dataloader, df_int_val.shape[0],
                'InternalValidation', epoch=epoch, fold=fold)
            testing_loss, testing_metric = self.eval_model(
                model, test_dataloader, df_test.shape[0],
                'Testing', epoch=epoch, fold=fold)
            ext_validation_loss, ext_validation_metric = self.eval_model(
                model, ext_val_dataloader, df_ext_val.shape[0],
                'ExternalValidation', epoch=epoch, fold=fold)

            # Comparison is by AUC-ROC
            if testing_metric > extreme_metric:
                extreme_metric = testing_metric
                if Config.save_models:
                    print('Saving model')
                    outfile_path = f'Models/{self.outcome}/C_{self.img_model_type}_F{fold}.pth'
                    torch.save(model, outfile_path)

            results = f'Training: {training_loss} | Validation: {validation_metric} | Testing: {testing_metric} | MT: {extreme_metric}'
            print(results)

            df_results = pd.DataFrame([
                self.outcome, self.img_model_type, fold,
                epoch + self.epoch_offset, training_loss,
                validation_loss, validation_metric,
                testing_loss, testing_metric,
                ext_validation_loss, ext_validation_metric]).T
            df_results.to_csv(
                f'ClassificationResults_{self.outcome}.csv',
                mode='a',
                header=False)

            # Patience
            performance_track.append(testing_metric)
            if len(performance_track) >= Config.patience:
                perf_eval = performance_track[(len(performance_track) - Config.patience):]
                if sorted(perf_eval, reverse=True) == perf_eval:  # Tracking AUROC
                    print(f'Patience threshold exceeded @E {epoch} @TP {perf_eval[0]} > {perf_eval[-1]}')
                    return

    def hammer_time(self):
        # One time split for internal / external validation
        df_int = self.df_patients.query(
            'SITENAME != @Config.ext_val_hospital').reset_index(drop=True)
        df_ext_val = self.df_patients.query(
            'SITENAME == @Config.ext_val_hospital').reset_index(drop=True)

        # Stratified cross val per group
        splits_file = f'{Config.file_splits}_{self.outcome}.pickle'
        if os.path.exists(splits_file):
            print('Using pickled splits')
            splits = pd.read_pickle(splits_file)
        else:
            splits = self.get_splits(df_int)
            pd.to_pickle(splits, splits_file)

        for fold, (train_idx, test_val_idx) in enumerate(splits):
            df_train = df_int.loc[train_idx]
            df_val_test = df_int.loc[test_val_idx]

            # Deduplicate for test set
            df_val_test = df_val_test.sort_values(['MRN', 'TIME_DELTA'])
            df_val_test = df_val_test.reset_index().groupby(['MRN', 'ECHODate']).first()
            df_val_test = df_val_test.drop_duplicates(subset=['IDENTIFIER'])
            df_val_test = df_val_test.reset_index().set_index('index')

            df_test, df_val = train_test_split(
                df_val_test,
                random_state=Config.random_state,
                shuffle=True,
                test_size=0.25,
                stratify=df_val_test['OUTCOME'])

            dict_dataframes = {
                'train': df_train,
                'test': df_test,
                'int_val': df_val,
                'ext_val': df_ext_val
            }

            # Train as usual - with awareness of folds and patience
            try:
                self.gaping_maw(dict_dataframes, fold)
            except KeyboardInterrupt:
                print('Continuing to next fold')
                continue
