DATA_PATH = None
CHECKPOINT_DIR  = None
def init(data_path: str, checkpoint_dir: str):
    global DATA_PATH, CHECKPOINT_DIR
    DATA_PATH = data_path
    CHECKPOINT_DIR = checkpoint_dir