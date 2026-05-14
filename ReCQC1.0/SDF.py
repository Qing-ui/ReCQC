import sys
from rdkit import Chem
import tkinter as tk
from tkinter import filedialog
import os
from USERINPUT import *


class File_path:
    def __init__(self, system_path=None, user_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.system_path_base = {
            'experiment_data': os.path.join(base_dir, 'DB', 'experiment.sdf'),
            'model_data': os.path.join(base_dir, 'DB', 'model.sdf'),
            'prediction_data': os.path.join(base_dir, 'DB', 'prediction.sdf')
        }
        self.system_path = system_path if system_path is not None else []
        self.user_path = user_path if user_path is not None else []

    def select_realdata_path(self):
        self.system_path.append(self.system_path_base['experiment_data'])
        return self.system_path

    def remove_realdata_path(self):
        if self.system_path_base['experiment_data'] in self.system_path:
            self.system_path.remove(self.system_path_base['experiment_data'])
        return self.system_path


    def select_model_path(self):
        self.system_path.append(self.system_path_base['model_data'])
        return self.system_path


    def remove_model_path(self):
        if self.system_path_base['model_data'] in self.system_path:
            self.system_path.remove(self.system_path_base['model_data'])


    def select_prediction_path(self):
        self.system_path.append(self.system_path_base['prediction_data'])
        return self.system_path

    def remove_prediction_path(self):
        if self.system_path_base['prediction_data'] in self.system_path:
            self.system_path.remove(self.system_path_base['prediction_data'])
        return self.system_path

    def select_user_path(self, user_path_item):
        self.user_path = []
        self.user_path.extend(user_path_item)

        return self.check_user_path_file(),self.user_path

    def check_user_path_file(self):
        errors = []  # 用于收集所有错误
        for index, user_path in enumerate(self.user_path, start=1):
            supplier = Chem.SDMolSupplier(user_path)
            for mol_index, mol in enumerate(supplier, start=1):
                if mol is None:
                    errors.append(f'In file {user_path}, the {mol_index} numerator is unreadable.')
                    continue
                    # 定义一个列表，包含要检查的所有属性名
                properties_to_check = ['ID','FW', 'Quaternaries', 'Tertiaries', 'Secondaries', 'Primaries']

                for prop in properties_to_check:
                    try:
                        prop_value = mol.GetProp(prop)
                        if prop == 'ID' and (prop_value is None or prop_value == ''):
                            # 对ID属性进行特殊处理，因为它不能为空
                            errors.append(f'In the file {user_path}, the ID attribute of the {mol_index} molecule is absent or empty.')
                        elif prop == 'FW' and (prop_value is None or prop_value == ''):
                            # 对FW属性进行特殊处理，因为它不能为空
                            errors.append(f'In the file {user_path}, the FW attribute of the {mol_index} molecule is absent or empty.')

                    except KeyError:
                        # 如果属性不存在，则记录错误
                        errors.append(f'In the file {user_path}, the {prop} attribute of the {mol_index} molecule does not exist. ')

        if errors:
            for error in errors:
                print(error)

        else:

            return "All documents are checked and there are no errors."


    def remove_user_path(self, user_path_item):
        if user_path_item in self.user_path:
            self.user_path.remove(user_path_item)
        return self.user_path

    def select_all_paths(self):
        return self.user_path + self.system_path


class Process_sdf_files:
    def __init__(self, file_paths_object,compound_shift_type_name,
                 min_weight = 0.0, max_weight = 3000.0, len_min_c_atom_num = 10, len_max_c_atom_num = 100):


        self.all_paths = file_paths_object
        self.compound_inf_dict = {}
        self.compound_shift_type_name = compound_shift_type_name
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_c_atom_num = len_min_c_atom_num
        self.max_c_atom_num = len_max_c_atom_num
        self.substructure_list = [
            # 常见芳香环
            'c1ccccc1',          # benzene
            'c1ccccc1C',         # toluene-like phenyl methyl
            'c1ccc(O)cc1',       # phenol
            'c1ccc(OC)cc1',      # anisole-like
            'c1ccc(C=O)cc1',     # benzaldehyde/aryl carbonyl
            'c1ccc(C(=O)O)cc1',  # benzoic acid-like
            'c1ccc(C(=O)N)cc1',  # benzamide-like
        
            # 常见脂肪环
            'C1CC1',             # cyclopropane
            'C1CCC1',            # cyclobutane
            'C1CCCC1',           # cyclopentane
            'C1CCCCC1',          # cyclohexane
            'C1CCCCCC1',         # cycloheptane
            'C1=CCCCC1',         # cyclohexene
            'C1=CCCC1',          # cyclopentene
        
            # 常见含氧环
            'C1CO1',             # oxirane
            'C1CCO1',            # oxetane
            'C1CCOC1',           # tetrahydrofuran
            'C1CCOCC1',          # tetrahydropyran
            'O1CCOCC1',          # 1,4-dioxane
            'O=C1OCCC1',         # gamma-lactone
            'O=C1OCCCC1',        # delta-lactone
        
            # 常见含氮环
            'C1CCNCC1',          # piperidine
            'C1CNCCN1',          # piperazine
            'C1COCCN1',          # morpholine
            'C1CCNC1',           # pyrrolidine
        
            # 常见芳香杂环
            'c1ccncc1',          # pyridine
            'c1cnccn1',          # pyrazine-like diazine
            'c1ncc[nH]1',        # imidazole
            'c1ccoc1',           # furan
            'c1ccsc1',           # thiophene
            'c1cc[nH]c1',        # pyrrole
            'c1noccc1',          # oxazole/isoxazole-like
            'c1nccs1',           # thiazole-like
        
            # 常见稠合环
            'c1ccc2ccccc2c1',    # naphthalene
            'c1ccc2[nH]ccc2c1',  # indole
            'c1ccc2ncccc2c1',    # quinoline
            'c1ccc2cnccc2c1',    # isoquinoline-like
            'O=c1oc2ccccc2cc1',  # coumarin-like
        ]
        self.process()
    def process(self):
        for path in self.all_paths:
            self.process_file(path)


    def process_file(self, path):
        supplier = Chem.SDMolSupplier(path)
        for mol in supplier:
            if mol is not None:
                self.process_molecule(mol)


    def process_molecule(self, mol):
        compound_id = self.get_compound_id(mol)
        fw = self.get_fw(mol)
        carbon_count = self.get_compound_c_atom_num(mol)
        if compound_id is None:
            return  # Skip if ID is not found
        elif fw is None:
            return  # Skip if FW is not found
        if self.min_weight <= fw <= self.max_weight and carbon_count in range(self.min_c_atom_num, self.max_c_atom_num + 1):
            compound_shift, compound_shift_type, compound_c_atom_num_list = self.extract_shift_data(mol, compound_id)
            ske_index_all_matches = self.get_skeleton_index(mol)
            self.add_shift_data(compound_id, compound_shift, compound_shift_type, compound_c_atom_num_list,ske_index_all_matches)

    def get_compound_c_atom_num(self, mol):
        carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        return carbon_count

    def get_compound_id(self, mol):
        try:
            return mol.GetProp('ID')
        except KeyError:
            return None
    def get_fw(self, mol):
        try:
            fw = float(mol.GetProp('FW'))
            return fw
        except:
            return None

    def get_skeleton_index(self, mol):
        index_all_matches = []
        for substructure_str in self.substructure_list:
            substructure = Chem.MolFromSmiles(substructure_str)
            matches = mol.GetSubstructMatches(substructure)

            for match in matches:
                index_all_matches.append(match)


        return index_all_matches






    def extract_shift_data(self, mol, compound_id):
        compound_shift = []
        compound_shift_type = []
        compound_c_atom_num_list = []

        for type_name in self.compound_shift_type_name:
            shift_str = self.get_shift_string(mol, compound_id, type_name)
            if shift_str:
                self.process_shift_string(shift_str, compound_c_atom_num_list, compound_shift, compound_shift_type,
                                          type_name)

        return compound_shift, compound_shift_type, compound_c_atom_num_list

    def get_shift_string(self, mol,compound_id ,type_name):
        errors = []
        try:
            return mol.GetProp(type_name)
        except KeyError:
            errors.append(f'Compound {compound_id} does not have a {type_name} property')
        if errors:
            print(errors)
            return False

    def process_shift_string(self, shift_str, compound_c_atom_num_list, compound_shift, compound_shift_type, type_name):
        error_list = []
        lines = shift_str.split('\n')
       #     # 检查是否有行数据，如果为空，则直接跳出处理
        if not lines or all(line.strip() == "" for line in lines):
            return
        for line in lines:
            line = line.strip()
            if line:  # 确保行不为空
                parts = line.split('\t')
                # 确保列表长度正确
                if len(parts) == 2:
                    try:
                        compound_c_atom_num_list.append(int(parts[0])-1)
                        compound_shift.append(float(parts[1]))
                        compound_shift_type.append(type_name)
                    except ValueError:
                        error_list.append(f'Error data row: {line}, data cannot be converted to integer or floating-point number')
                else:
                    error_list.append(f'Error data row: {line}, which should contain two tab-separated fields')
        if error_list:
            return error_list,exit(1)

    def add_shift_data(self, compound_id, compound_shift, compound_shift_type, compound_c_atom_num_list,ske_index_all_matches):

        compound_inf = [compound_c_atom_num_list,compound_shift, compound_shift_type, ske_index_all_matches]
        compound_inf_dict = {compound_id: compound_inf}
        self.compound_inf_dict.update(compound_inf_dict)




class Add_user_path:
    def __init__(self):
        self.user_path = []

    def select_user_path(self):
        root = tk.Tk()
        root.withdraw()  # 隐藏根窗口
        # 弹出文件选择对话框
        file_path = filedialog.askopenfilename()

        # 检查用户是否选择了一个文件
        if file_path:
            # 检查文件后缀是否为sdf
            if os.path.splitext(file_path)[1].lower() == '.sdf':
                self.user_path.append(file_path)  # 将文件路径添加到列表中
                self.user_path = list(set(self.user_path))  # 去重
                print(f"The file path has been added: {file_path}")
            else:
                print(f"The suffix of file {file_path} is not sdf and has been ignored.")
        print(self.user_path)

                # 销毁临时创建的Tkinter根窗口
        root.destroy()
        return self.user_path


# file_paths = File_path()
# user_path_init = Add_user_path()
# user_path=user_path_init.select_user_path()
# print(user_path)
# file_paths.select_user_path(user_path)
# # 选择模型路径、用户路径和实验数据路径
# file_paths.select_model_path()
# # 查看当前选择的所有路径
# all_paths = file_paths.select_all_paths()
# rd=ProcesInputdata('','1,2,3,4','1,2,3','1,2,3,4',)
# ctype=rd.match_type_state
# print(ctype)
# b=Process_sdf_files(all_paths,ctype)
# b.process()
# print(b.compound_inf_dict)












