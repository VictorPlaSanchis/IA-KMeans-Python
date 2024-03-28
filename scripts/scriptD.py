import sys
import os

def main(dataset_name):
    # Iterar sobre K de 2 a 10
    for D in [2,10,20,30,40,50,60,70,80,90]:
        # Iterar sobre I de 0 a 4
        for I in range(10):
            print(f"py main.py {10} {dataset_name+str(D)}> ./outputs/dW/output-D{D}-I{I}.log")
            # Ejecutar el comando y guardar la salida en un archivo de registro
            os.system(f"py main.py {10} {dataset_name+str(D)} > ./outputs/dW/output-D{D}-I{I}.log")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset_name>")
        sys.exit(1)
    dataset_name = sys.argv[1]
    main(dataset_name)
