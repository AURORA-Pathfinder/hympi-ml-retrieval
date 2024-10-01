import pickle
files_to_redo = []
with open("redo.pkl", 'wb') as f:
    pickle.dump(files_to_redo, f)
