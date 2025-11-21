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
        self.substructure_list = ['O=C1OC2=C(C=CC=C2)C(C3=CC=CC=C3)=C1',
        'C12CCCCC1CCC3C2=CCC4C5CCCCC5CCC43',
        'C[C@@](C=C)(C1)CCC2=C1CCC3[C@@]2(CCC[C@]3(CO)C)C',
        'C1/C=C/CC/C=C/CCC1',
        'O=C(C1=CC=CC=C1O2)C=C2C3=CC=CC=C3',
        'O=C1C=CCC2=CCC3C4CCCC4CCC3C12',
        'O=C1C2=C(C(C3=CC=CC=C13)=O)C=CC=C2',
        'O=C1CCC2CCC3C(CC(O3)=O)CC12',
       'O=C1C=C2CC3C(C(CC=C3)=O)CC2O1',
        'C12CCC3CCCCC3C1CCC4C2CCC4',
        'C1(/C=C/C2=CC=CC=C2)=CC=CC=C1'
        'C12CCC3C(C1CCCC2)CCC4C3CCC5CCCC54',
        'C12CCC/C=C\CCCC1C=CCC2',
        'O=C([C@@H]1CC2=CC=CC=C2)OC[C@@H]1CC3=CC=CC=C3',
        'C12CC[C@@]34C[C@@H](CC4)CCC3C1CCCC2',
        'C12CCC3C(CC1=CCC4CCCCC24)CCC5CCCCC53',
        'C=C1CCC2C(OC(C2)=O)C3CCCC13',
        'O=C1[C@@]23C=CCC2CC=C[C@@H]1C4CC4CC3',
        'O=C1C2CC[C@@]3(CC4=C(COC4)CCC3C5=O)O[C@@]65OC7[C@H](OC(C7)=O)C(C26)C1',
        'C12CC1C3CCCC3CCC2',
        'O=C1CC2C/C=C/CC/C=C/CC/C=C/CC2O1',
        'CCCC1=CC(OC2=C1C3=C(C=CC(C)(O3)C)C4=C2CCCO4)=O',
        '[H][C@@]12C[C@@H]([C@@]3([C@H](CCC[C@@]3(OC2(C)C)C1)OC(C4=CC=CC=C4)=O)C)OC(/C=C/C5=CC=CC=C5)=O',
       'C12CCCC1C3CCCC3C2',
        'C12CCC3=CC[C@H]4CC[C@]3(C1CCCC2)C4',
        'CC1(C)CCCC2C3COCC3=CCC21',
        'C12CCC3C=CCC31CCC2',
        'O=C(CC1=CC=CC=C12)CCCCCCCCOC2=O',
        'O=C1C2C(CC[C@H]2C(C)C)CCCC1',
        'C=C(C)CCCCC1CCC(C)=CC1=O',
        'C12CO[C@@H](C1CO[C@H]2C3=CC=CC=C3)C4=CC=CC=C4',
        'O=C1OC2C3CCCC3CCCC2C1',
        'C/C(CCC1CCCC2C=CCCC21)=C\C(OC)=O',
        'C[C@@]1(C=C)CC2C([C@@]3(CCC[C@H](C3CC2=O)C)C)CC1',
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












