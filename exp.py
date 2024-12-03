import core

model = core.Model("FullyConnected", "FullyConnected.yaml")
print(model)
model.train_model("train_FullyConnected.yaml")