# import torch
#
# def auto_encoder_model(in_features: int, hidden_size: int = 8):
#     return torch.nn.Sequential(
#         torch.nn.Linear(in_features, hidden_size),
#         torch.nn.BatchNorm1d(hidden_size),
#         torch.nn.GELU(),
#         torch.nn.Linear(hidden_size, int(hidden_size / 2)),
#         torch.nn.GELU(),
#         torch.nn.Linear(int(hidden_size / 2), hidden_size),
#         torch.nn.BatchNorm1d(hidden_size),
#         torch.nn.GELU(),
#         torch.nn.Linear(hidden_size, in_features),
#         torch.nn.GELU()
#     )
#
# class AutoEncoderInterpreter():
#     def __init__(self, state_dict, threshold, in_features=15, hidden_size=8):
#         self.model = auto_encoder_model(in_features=in_features, hidden_size=hidden_size)
#         self.model.load_state_dict(state_dict)
#         self.threshold = threshold
#
#     def predict(self, x):
#         test_data = torch.utils.data.TensorDataset(
#             torch.from_numpy(x).type(torch.float)
#         )
#         data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
#
#         all_predictions = torch.tensor([])  # .cuda()
#
#         self.model.eval()
#         with torch.no_grad():
#             ae_loss = torch.nn.MSELoss(reduction="sum")
#             for idx, (batch_x,) in enumerate(data_loader):
#                 batch_x = batch_x  # .cuda()
#                 model_predictions = self.model(batch_x)
#
#                 model_predictions = ae_loss(model_predictions, batch_x).unsqueeze(0)  # unsqueeze as batch_size set to 1
#                 all_predictions = torch.cat((all_predictions, model_predictions))
#
#         # all_predictions = all_predictions.tolist()
#         all_predictions = (all_predictions > self.threshold).type(torch.long)
#         return all_predictions.flatten()
