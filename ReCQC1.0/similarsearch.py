from  rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity
import  os

class SimilarSearch(object):
    def __init__(self, smiles, sdf_folder_path, threshold=0.8, radius=3):
        self.smiles = smiles
        self.sdf_folder_path = sdf_folder_path
        self.threshold = threshold
        self.radius = radius
        self.ref_mol = Chem.MolFromSmiles(self.smiles)
        if not self.ref_mol:
            raise ValueError("Invalid SMILES string")
        self.ref_fp = AllChem.GetMorganFingerprintAsBitVect(self.ref_mol, radius=self.radius, useFeatures=True)

    def get_similars(self):
        similar_molecules = []

        if not os.path.isdir(self.sdf_folder_path):
            print(f'No such directory: {self.sdf_folder_path}')
            return

        for root, dirs, files in os.walk(self.sdf_folder_path):
            for file in files:
                if file.endswith(".sdf"):
                    sdf_path = os.path.join(root, file)
                    supplier = Chem.SDMolSupplier(sdf_path)
                    for mol in supplier:
                        if mol is not None:
                            mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, useFeatures=True)
                            similarity = FingerprintSimilarity(self.ref_fp, mol_fp)
                            if similarity >= self.threshold:
                                similar_molecules.append((mol, similarity))
        if similar_molecules:
            output_sdf_path = "similar_molecules.sdf"
            with Chem.SDWriter(output_sdf_path) as writer:
                for mol, similarity in similar_molecules:
                    writer.write(mol)
            print(f"Successfully written {len(similar_molecules)} similar molecules to {output_sdf_path}")
        else:
            print("No similar molecules found.")

class GetSubstructures(object):
    def __init__(self, substructure_smiles, sdf_folder_path):
        self.substructure_smiles = substructure_smiles
        self.sdf_folder_path = sdf_folder_path
        self.substructure_mol = Chem.MolFromSmiles(self.substructure_smiles)
        if not self.substructure_mol:
            raise ValueError("Invalid SMILES string for substructure")

    def get_substructures_sdf(self):
        molecules_with_substructure = []
        # 遍历文件夹中的所有文件
        for root, dirs, files in os.walk(self.sdf_folder_path):
            for file in files:
                if file.endswith(".sdf"):  # 只处理SDF文件
                    sdf_path = os.path.join(root, file)
                    supplier = Chem.SDMolSupplier(sdf_path)
                    for mol in supplier:
                        if mol is not None:
                            if mol.HasSubstructMatch(self.substructure_mol):
                                molecules_with_substructure.append(mol)

        if molecules_with_substructure:
            output_sdf_path = "molecules_with_substructure.sdf"
            with Chem.SDWriter(output_sdf_path) as writer:
                for mol in molecules_with_substructure:
                    writer.write(mol)
            print(f"Successfully written {len(molecules_with_substructure)} molecules with substructure to {output_sdf_path}")
        else:
            print("No molecules with the specified substructure found.")








