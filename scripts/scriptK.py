import sys
import os

def main(dataset_name):
    # Iterar sobre K de 2 a 10
    for K in range(2,25):
        # Iterar sobre I de 0 a 4
        for I in range(10):
            print(f"py main.py {K} {dataset_name}> ./outputs/kW/output-K{K}-I{I}.log")
            # Ejecutar el comando y guardar la salida en un archivo de registro
            os.system(f"py main.py {K} {dataset_name} > ./outputs/kW/output-K{K}-I{I}.log")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset_name>")
        sys.exit(1)
    dataset_name = sys.argv[1]
    main(dataset_name)
