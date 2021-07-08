#!/bin/python


class Config:
    echo_root = '/disks/Work/Databases/EchoReports'
    df_files = 'Data/EchoReports.pickle'  # This file must exist for starting NLP parsing
    threads = 5  # Most efficient - slows down before or after 5

    # Result collation of NLP and manual review
    collated_df = 'Data/ExtractedEchoParams.pickle'  # This file must exist at the time of ECG joining
    dir_nlp_results = 'NLPResults'
    review_packages = 5
    files_per_outcome = 15
    random_state = 42

    # ECG acquisition and plotting
    ecg_echo_timedelta = 7
    file_ecg_query = 'Data/ECGQuery.pickle'
    file_ecg_echo = 'Data/ECGEchoJoin.pickle'

    file_scaling = 'Data/LeadScalingValvular.pickle'
    file_ecg_metrics = 'Data/DerivedMetricsValvular.pickle'

    dir_ecg_xml = '../ECGs/XMLFiles'
    dir_ecg_plots = '../ECGs/PlottedECGsValvular'

    dir_ecg_xml_ps = '../ECGs/PreScreeningXMLFiles/'
    dir_ecg_plots_ps = '../ECGs/PreScreeningPlottedECGs/'

    # ECG XMl specific
    sampling_freq = 500  # Constant for all
    perform_filtering = True
    attenuate = True  # Only carried out in case divide_into_beats is False
    plot_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Training related
    debug = False

    random_state = 42
    dropout_rate = False

    epochs = 301  # Make sure this is odd
    resize = True
    save_models = True

    # Workable batch sizes with AMP
    # Efficientnet B5 (456): 40
    # Resnet50 (456): 128
    batch_size = 150

    # External validation - by facility
    ext_val_hospital = 'ST.LUKE\'S-ROOSEVELT HOSPITAL (S)'

    # Cross validation folds
    file_splits = 'Data/Splits'  # Store split indices here to speed things up
    cross_val_folds = 10
    patience = 3  # Epochs for which performance must go down during cross-val for training to be discontinued
