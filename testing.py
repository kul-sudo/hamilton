from keras.models import load_model
from numpy import array
from config import *
from json import load

model = load_model("epoch=13 val_accuracy=0.971 val_loss=0.077.keras")

with open("testing/hamiltonian_graphs.json", "r") as file:
    hamiltonian_graphs_data = load(file)

with open("testing/non_hamiltonian_graphs.json", "r") as file:
    non_hamiltonian_graphs_data = load(file)

if hamiltonian_graphs_data["info"] != non_hamiltonian_graphs_data["info"]:
    print("The info has to be identical in both files.")
    exit(0)

NODES_N = hamiltonian_graphs_data["info"]["nodes_n"]
GRAPHS_N = hamiltonian_graphs_data["info"]["graphs_n"]

print(f"{NODES_N=}\n{GRAPHS_N=}\n")

hamiltonian_graphs, non_hamiltonian_graphs = (
    hamiltonian_graphs_data["graphs"],
    non_hamiltonian_graphs_data["graphs"],
)

results_hamiltonian = 0
results_non_hamiltonian = 0

BOUND = 0.5

predictions_sum = 0
for i in range(GRAPHS_N):
    print(f"\r{round((i+1)/GRAPHS_N*100)}%", end="")
    predictions_sum += (
        prediction := model.predict(array([hamiltonian_graphs[i]]), verbose=0)[0][0]
    )
    results_hamiltonian += int(prediction >= BOUND)
print(f"\rHamiltonian accuracy: {results_hamiltonian/GRAPHS_N:.2f}.")
print(f"Average prediction: {predictions_sum/GRAPHS_N:.2f}")
print()

predictions_sum = 0
for i in range(GRAPHS_N):
    print(f"\r{round((i+1)/GRAPHS_N*100)}%", end="")
    predictions_sum += (
        prediction := model.predict(array([non_hamiltonian_graphs[i]]), verbose=0)[0][0]
    )
    results_non_hamiltonian += int(prediction < BOUND)
print(f"\rNon-hamiltonian accuracy: {results_non_hamiltonian/GRAPHS_N:.2f}.")
print(f"Average prediction: {predictions_sum/GRAPHS_N:.2f}")
print()

print(
    f"Total accuracy: {(results_hamiltonian + results_non_hamiltonian)/(2*GRAPHS_N):.2f}."
)
