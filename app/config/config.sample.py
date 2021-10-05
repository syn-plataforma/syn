class Config(object):
    DEBUG = True
    TESTING = False
    API_VERSION = 'v1'


class DevelopmentConfig(Config):
    DEBUG = False

    WORK_ENVIRONMENT = 'local'
    ARCHITECTURE = 'codebooks'

    MONGO_DATA_USER = '******'
    MONGO_DATA_PASSWORD = '******'
    MONGO_DATA_HOST = 'localhost'
    MONGO_DATA_PORT = 27017
    MONGO_DATA_AUTH_SOURCE = '******'
    MONGO_DATA_AUTH_MECHANISM = '******'
    MONGO_DATA_BUGZILLA_DATABASE = 'bugzilla'
    MONGO_DATA_NETBEANS_DATABASE = 'netBeans'
    MONGO_DATA_OPENOFFICE_DATABASE = 'openOffice'
    MONGO_DATA_COLLECTION = 'clear'
    MONGO_DUPLICATES_COLLECTION = 'duplicates'
    MONGO_DATA_ASSIGNATION_DATABASE = 'assignation'
    MONGO_DATA_OCCUPATION_COLLECTION = 'occupation'
    MONGO_DATA_ADEQUACY_COLLECTION = 'adecuation'

    MONGO_API_USER = '******'
    MONGO_API_PASSWORD = '******'
    MONGO_API_HOST = 'localhost'
    MONGO_CODEBOOKS_API_HOST = 'localhost'
    MONGO_API_PORT = 27017
    MONGO_API_AUTH_SOURCE = '******'
    MONGO_API_AUTH_MECHANISM = '******'
    MONGO_API_DATABASE = 'syn_rest_api'
    MONGO_API_TRAINING_PARAMETERS_COLLECTION = 'training_parameters'
    MONGO_API_USERS_COLLECTION = 'users'

    JWT_SECRET_KEY = '******'
    JWT_ACCESS_TOKEN_EXPIRES = 60

    SWAGGER_URL = '/api/docs'

    SECRET_KEY = '******'
    SECURITY_PASSWORD_SALT = '******'
    MAIL_DEFAULT_SENDER = '******'
    MAIL_SERVER = '******'
    MAIL_PORT = 465
    MAIL_USERNAME = '******'
    MAIL_PASSWORD = '******'
    MAIL_USE_TLS = False
    MAIL_USE_SSL = True

    MODEL_DUMP_PATH = 'C:\\******\\host-mounted-volumes\\syn\\dump\\'
    EVALUATE_DATA_PATH = 'C:\\******\\host-mounted-volumes\\api\\evaluate_data\\'

    DUPLICATES_THRESHOLD = 0.5

    TASKS_DATABASE_NAME = 'tasks'
    EXPERIMENTS_COLLECTION_NAME = 'experiments'

    # Stanford CoreNLP server
    CORENLP_SERVER_HOST = "******"
    CORENLP_SERVER_PORT = 9000

    EMBEDDING_MONGODB_MAX_NUM_TOKENS = 150

    DATA_PATH = 'C:\\******\\host-mounted-volumes\\syn\\data\\'

    # DyNet config
    DYNET_USE_GPU = 0
    DYNET_SEED = 230
    DYNET_MEM = 512
    DYNET_GPUS = 0
    DYNET_AUTOBATCH = 0

    STANFORD_CORENLP_PATH = 'C:\\************\\lib\\stanford-corenlp'
