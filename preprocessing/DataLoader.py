from enum import Enum, auto
import numpy as np

# Enum class that contains the list of all available dataests in the full_data defined above
class DataName(Enum):
    HyMPI_Composite = auto()
    HyMPI_a = auto()
    HyMPI_b = auto()
    HyMPI_c = auto()
    HyMPI_d = auto()
    HyMPI_w = auto()
    HyMPI_1 = auto()
    CoSMIR_H = auto()
    BSL = auto()
    BSL_Depth = auto()
    ATMS = auto()

    Labels_Table = auto()
    LatLon = auto()
    PBLH = auto()
    Surface_Pressure = auto()
    Surface_Temperature = auto()
    DateAndTime = auto()

    Labels_Scalar = auto()
    Pressure = auto()
    Height = auto()
    Temperature = auto()
    Water_Vapor = auto()
    Ozone_Density = auto()

# Class allowing the loading and transforming of a dataset
class DataLoader():
    def __init__(self, file_path: str, dry_run: bool = False, verbose: bool = True):
        self.dry_run = dry_run
        self.file_path = file_path
        self.requested = []

        if verbose:
            print(f"Started dataloader with {file_path}")

        # TODO should be fine given its a npz load
        #  but check anyway
        self.full_data = np.load(self.file_path)


    # Returns a single dataset given its name
    def get_data(self, dataName: DataName) -> np.ndarray:
        if isinstance(dataName, str):
            dataName = DataName[dataName]

        # TODO see if this helps speed
        # full_data = np.load(self.file_path)

        # TODO if multiple dataloaders this won't work
        self.requested.append(dataName)

        if self.dry_run:
            print(f"Dry running no data returned for {dataName}!")
            return np.zeros((1000, 22))

        match dataName:
            case DataName.HyMPI_Composite: return self.full_data['hs']
            case DataName.HyMPI_a: return self.full_data['ha']
            case DataName.HyMPI_b: return self.full_data['hb']
            case DataName.HyMPI_c: return self.full_data['hc']
            case DataName.HyMPI_d: return self.full_data['hd']
            case DataName.HyMPI_w: return self.full_data['hw']
            case DataName.HyMPI_1: return self.full_data['h1']
            case DataName.CoSMIR_H: return self.full_data['ch']
            case DataName.BSL: return self.full_data['bs']
            case DataName.BSL_Depth: return self.full_data['bs_depth']
            case DataName.ATMS: return self.full_data['mh']

            case DataName.Labels_Scalar: return self.full_data['labels_scalar']
            case DataName.LatLon: return self.full_data['labels_scalar'][:, 1:3]
            case DataName.PBLH: return self.full_data['labels_scalar'][:, 0]
            case DataName.Surface_Pressure: return self.full_data['labels_scalar'][:, 3]
            case DataName.Surface_Temperature: return self.full_data['labels_scalar'][:, 4]
            case DataName.DateAndTime: return self.full_data['labels_scalar'][:, 5]

            case DataName.Labels_Table: return self.full_data['labels_table']
            case DataName.Pressure: return self.full_data['labels_table'][:,:,0]
            case DataName.Height: return self.full_data['labels_table'][:,:,1]
            case DataName.Temperature: return self.full_data['labels_table'][:,:,2]
            #case DataName.Water_Vapor: return self.full_data['labels_table'][:,36:,3]
            case DataName.Water_Vapor: return self.full_data['labels_table'][:,:,3]
            case DataName.Ozone_Density: return self.full_data['labels_table'][:,:,4]
            case _: print("No match for " + str(dataName) +" found in file: " + self.file_path)

            # TODO error handling
