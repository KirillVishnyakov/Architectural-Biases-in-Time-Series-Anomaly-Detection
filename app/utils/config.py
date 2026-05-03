from pathlib import Path

class Config:
    def __init__(
        self, 
        root_dir, 
        app_dir, 
        data_dir, 
        artifacts_dir,
        weights_dir, 
        training_results_dir
    ):

        self.root = Path(root_dir)

        self.app_dir = self.root / app_dir
        self.data_dir = self.app_dir / data_dir

        self.artifacts_dir = self.root / artifacts_dir
        self.weights_dir = self.artifacts_dir / weights_dir
        self.training_results_dir = self.artifacts_dir / training_results_dir

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.training_results_dir.mkdir(parents=True, exist_ok=True)
