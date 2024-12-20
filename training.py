import matplotlib.pyplot as plt
from config import *
from numpy import array
from callbacks import *
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from model import create_model
from sys import exit
from json import load

with open("hamiltonian_graphs.json", "r") as file:
    hamiltonian_graphs_data = load(file)

with open("non_hamiltonian_graphs.json", "r") as file:
    non_hamiltonian_graphs_data = load(file)

if hamiltonian_graphs_data["info"] != non_hamiltonian_graphs_data["info"]:
    print("The info has to be identical in both files.")
    exit(0)

NODES_N = hamiltonian_graphs_data["info"]["nodes_n"]
GRAPHS_N = hamiltonian_graphs_data["info"]["graphs_n"]

print(f"{NODES_N=}\n{GRAPHS_N=}")

TRAIN_N = GRAPHS_N - VALIDATION_N

for graphs_data in (hamiltonian_graphs_data, non_hamiltonian_graphs_data):
    graphs = graphs_data["graphs"]
    for i in range(0, len(graphs)):
        graphs[i] = [[int(x) for x in y] for y in graphs[i]]

x_train = array(
    hamiltonian_graphs_data["graphs"][:TRAIN_N]
    + non_hamiltonian_graphs_data["graphs"][:TRAIN_N]
)
y_train = array([1] * TRAIN_N + [0] * TRAIN_N)

x_validate = array(
    hamiltonian_graphs_data["graphs"][TRAIN_N:]
    + non_hamiltonian_graphs_data["graphs"][TRAIN_N:]
)
y_validate = array([1] * VALIDATION_N + [0] * VALIDATION_N)

model = create_model(NODES_N)
model.summary()
optimizer = SGD(learning_rate=0.01)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

model_checkpoint_callback = ModelCheckpoint(
    filepath="checkpoints/epoch={epoch:02d} val_accuracy={val_accuracy:.3f}"
    " val_loss={val_loss:.3f}.keras",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
    save_freq="epoch",
)
history_saver = HistorySaver()
history = model.fit(
    x_train,
    y_train,
    epochs=10000,
    callbacks=[history_saver, model_checkpoint_callback],
    validation_data=(x_validate, y_validate),
    batch_size=32,
)
