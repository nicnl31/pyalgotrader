import torch
from torch import nn, Tensor

from typing import Any, Dict

from freqtrade.freqai.base_models.BasePyTorchClassifier import \
	BasePyTorchClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import \
	(DefaultPyTorchDataConvertor, PyTorchDataConvertor)
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchModelTrainer


class CLSTM(nn.Module):
	def __init__(self, n_features, n_obs):
		super(CLSTM, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv1d(n_features, 64, 1),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Dropout(0.25)
		)
		self.conv2 = nn.Sequential(
			nn.Conv1d(64, 32, 1),
			nn.BatchNorm1d(32),
			nn.ReLU(),
			nn.Dropout(0.25)
		)
		self.conv3 = nn.Sequential(
			nn.Conv1d(32, 32, 1),
			nn.BatchNorm1d(32),
			nn.ReLU(),
			nn.Dropout(0.25)
		)
		self.conv4 = nn.Sequential(
			nn.Conv1d(32, 8, 1),
			nn.BatchNorm1d(8),
			nn.ReLU(),
			nn.Dropout(0.25)
		)
		self.conv5 = nn.Sequential(
			nn.Conv1d(8, 1, 1),
			nn.BatchNorm1d(1),
			nn.ReLU(),
			nn.Dropout(0.25)
		)
		self.lstm = nn.Sequential(
			nn.LSTM(input_size=1, hidden_size=150, num_layers=1),
			nn.BatchNorm1d(1),
			nn.ReLU()
		)
		self.fc = nn.Sequential(
			nn.Linear(n_features*n_obs, 200),
			nn.Linear(200, 100),
			nn.Linear(100, 100),
			nn.Linear(100, 100),
			nn.Linear(100, 100),
			nn.Linear(100, 100),
			nn.Linear(100, 100),
			nn.Linear(100, 100),
			nn.Linear(100, 1)
		)

	def forward(self, x: Tensor) -> Tensor:
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.lstm(x)
		x = self.fc(x)

		return x


class CLSTMClassifier(BasePyTorchClassifier):
	@property
	def data_convertor(self) -> PyTorchDataConvertor:
		return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)
		config = self.freqai_info.get("model_training_parameters", {})
		self.learning_rate: float = config.get("learning_rate", 1e-3)
		self.model_kwargs: Dict[str, Any] = config.get("model_kwargs", {})
		self.trainer_kwargs: Dict[str, Any] = config.get("trainer_kwargs", {})

	def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
		"""
		:param data_dictionary:
			the dictionary holding all data for train, test, labels, weights
		:param dk: The datakitchen object for the current coin/model
		"""
		# Get class names
		class_names = self.get_class_names()

		# convert label columns to int columns for the model
		self.convert_label_column_to_int(
			data_dictionary=data_dictionary,
			dk=dk,
			class_names=class_names
		)

		# Declare number of features and observations
		n_features = data_dictionary["train_features"].shape[-1]
		n_obs = data_dictionary["train_features"].shape[0]

		# Set up the CLSTM parameters
		model = CLSTM(n_features=n_features, n_obs=n_obs)

		# Go to CUDA device if available, else CPU
		model.to(self.device)

		optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
		criterion = nn.CrossEntropyLoss()
		init_model = self.get_init_model(dk.pair)

		# Initialise the trainer
		trainer = PyTorchModelTrainer(
			model=model,
			optimizer=optimizer,
			criterion=criterion,
			model_meta_data={"class_names": class_names},
			device=self.device,
			init_model=init_model,
			data_convertor=self.data_convertor,
			**self.trainer_kwargs,
		)

		# Fit and return the trainer
		trainer.fit(data_dictionary=data_dictionary, splits=self.splits)

		return trainer
