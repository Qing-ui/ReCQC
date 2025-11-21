import os
from  rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import  pandas as pd
import  tkinter as tk
from tkinter import filedialog


class DBOrganization(object):
  def __init__(self,sdf_excel_obj,first_id = 1):
      self.sdf_excel_obj = sdf_excel_obj
      self.first_id = first_id
      #self.write_sdf()


  def is_directory(self):
      error = []
      if os.path.isdir(self.sdf_excel_obj):
          return True
      else:
          error.append('The path is not a file directory')
      return error

  def check_sdf_excel(self):
      errors = []  # 使用复数形式来表示可能存储多个错误
      sdf_files = [f for f in os.listdir(self.sdf_excel_obj) if f.endswith('.sdf')]
      excel_extensions = ['.xlsx', '.xls']
      if sdf_files == []:
          errors.append('No .sdf')

          for sdf_file in sdf_files:
              sdf_filename = os.path.splitext(sdf_file)[0]
              folder_path = self.sdf_excel_obj

              for filename in os.listdir(folder_path):
                  name, ext = os.path.splitext(filename)
                  if name == sdf_filename and ext in excel_extensions:
                      return True
                  else:
                      errors.append('No matching Excel file found for the SDF file')
          if errors != []:
              for error in errors:
                  print(error)
          else:
             return errors

  def check_sdf_mol(self):
      errors = []
      sdf_files = [f for f in os.listdir(self.sdf_excel_obj) if f.endswith('.sdf')]
      for sdf_file in sdf_files:
          mol_count = 0
          file_path = os.path.join(self.sdf_excel_obj, sdf_file)
          supplier = Chem.SDMolSupplier(file_path)
          for mol in supplier:
              if mol:
                 mol_count += 1
          if mol_count != 1:
              errors.append(f'Not one mol in SDF file: {sdf_file}')
          if errors != []:
              for error in errors:
                  print(error)
          else:
              return errors

  def write_sdf(self):
      output_sdf_path = '13c_generated.sdf'
      output_sdf = Chem.SDWriter(output_sdf_path)
      sdf_files = [f for f in os.listdir(self.sdf_excel_obj) if f.endswith('.sdf')]
      for sdf_file in sdf_files:
          sdf_file_path = os.path.join(self.sdf_excel_obj, sdf_file)
          # 构造与SDF文件同名的Excel文件路径
          excel_file_path = os.path.splitext(sdf_file_path)[0] + '.xlsx' or '.xls'

          if os.path.exists(excel_file_path):

              supplier = Chem.SDMolSupplier(sdf_file_path)

              # 读取Excel文件
              df = pd.read_excel(excel_file_path)
              data_lines = []
              result_string = ''
              Primaries_list = []
              Secondries_list = []
              Tertiaries = []
              Quaternaries = []
              all_c_dict = {}
              for index, row in df.iloc[0:].iterrows():
                  second_column = int(row[1])
                  third_column = row[2]#根据情况改一下列数，位移列
                  all_c_dict.update({second_column: third_column})
                  combined_data = f"{index}[{str(second_column)}]\t{str(third_column)}"
                  data_lines.append(combined_data)
                  result_string = '\n'.join(data_lines)
              for mol in supplier:
                if mol is not None:
                  print(mol)
                  self.first_id += 1
                  mol.SetProp('ID',str(self.first_id) )
                  mol.SetProp('13C_DATA', result_string)
                  for atoms in range(mol.GetNumAtoms()):
                      atom = mol.GetAtomWithIdx(atoms)
                      symbol = atom.GetSymbol()
                      if symbol == 'C'or symbol == 'c':
                          if int(atoms)+1 in all_c_dict.keys():
                              num_hydrogens = atom.GetTotalNumHs()
                              if num_hydrogens == 3:
                                  primaries_data = f"{str(int(atoms)+1)}\t{str(all_c_dict[int(atoms)+1])}"
                                  Primaries_list.append(primaries_data)
                              if num_hydrogens == 2:
                                  secondaries_data = f"{str(int(atoms)+1)}\t{str(all_c_dict[int(atoms)+1])}"
                                  Secondries_list.append(secondaries_data)
                              if num_hydrogens == 1:
                                  tertiary_data = f"{str(int(atoms)+1)}\t{str(all_c_dict[int(atoms)+1])}"
                                  Tertiaries.append(tertiary_data)
                              if num_hydrogens == 0:
                                  quaternary_data = f"{str(int(atoms)+1)}\t{str(all_c_dict[int(atoms)+1])}"
                                  Quaternaries.append(quaternary_data)
                  quaternary_string = '\n'.join(Quaternaries)
                  tertiary_string = '\n'.join(Tertiaries)
                  secondaries_string = '\n'.join(Secondries_list)
                  primaries_string = '\n'.join(Primaries_list)
                  mol_weight = Chem.rdMolDescriptors.CalcExactMolWt(mol)
                  mol.SetProp('FW', str(mol_weight))
                  fomular = Chem.rdMolDescriptors.CalcMolFormula(mol)
                  mol.SetProp('Formula', str(fomular))
                  mol.SetProp('Quaternaries', quaternary_string)
                  mol.SetProp('Tertiaries', tertiary_string)
                  mol.SetProp('Secondaries', secondaries_string)
                  mol.SetProp('Primaries', primaries_string)
                  output_sdf.write(mol)
          else:
              print('Please cheek')
      output_sdf.close()



class Add_file_path:
    def __init__(self):
        self.file_path = ''

    def select_user_path(self):
        errors = []
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.file_path = folder_path
        else:
            errors.append('The file is not a  folder')

        root.destroy()
        return self.file_path

    def delete_user_path(self):
        self.file_path = ''
#
# folder_path = Add_file_path().select_user_path()
# DBOrganization(sdf_excel_obj=folder_path)



