from enum import Enum, auto
from numpy import ndarray, load

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
    def __init__(self, file_path: str):
        self.file_path = file_path

    # Returns a single dataset given its name
    def get_data(self, dataName: DataName) -> ndarray:
        if isinstance(dataName, str):
            dataName = DataName[dataName]

        full_data = load(self.file_path)

        match dataName:
            case DataName.HyMPI_Composite: return full_data['hs']
            case DataName.HyMPI_a: return full_data['ha']
            case DataName.HyMPI_b: return full_data['hb']
            case DataName.HyMPI_c: return full_data['hc']
            case DataName.HyMPI_d: return full_data['hd']
            case DataName.HyMPI_w: return full_data['hw']
            case DataName.HyMPI_1: return full_data['h1']
            case DataName.CoSMIR_H: return full_data['ch']
            case DataName.BSL: return full_data['bs']
            case DataName.BSL_Depth: return full_data['bs_depth']
            case DataName.ATMS: return full_data['mh']

            case DataName.Labels_Scalar: return full_data['labels_scalar']
            case DataName.LatLon: return full_data['labels_scalar'][:, 1:3]
            case DataName.PBLH: return full_data['labels_scalar'][:, 0]
            case DataName.Surface_Pressure: return full_data['labels_scalar'][:, 3]
            case DataName.Surface_Temperature: return full_data['labels_scalar'][:, 4]
            case DataName.DateAndTime: return full_data['labels_scalar'][:, 5]

            case DataName.Labels_Table: return full_data['labels_table']
            case DataName.Pressure: return full_data['labels_table'][:,:,0]
            case DataName.Height: return full_data['labels_table'][:,:,1]
            case DataName.Temperature: return full_data['labels_table'][:,:,2]
            case DataName.Water_Vapor: return full_data['labels_table'][:,:,3]
            case DataName.Ozone_Density: return full_data['labels_table'][:,:,4] 
        
        print("No match for " + str(dataName) +" found in file: " + self.file_path)