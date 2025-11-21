from  rdkit import Chem
from rdkit.Chem import AllChem,rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from SDF import File_path,Process_sdf_files,Add_user_path
from USERINPUT import ProcesInputdata,ProductMatchData
from MATCHINSTITUTE import ScoreInstitution,ProductBestCompare
import pandas as pd
from itertools import islice
from PIL import Image
import io
import matplotlib.pyplot as plt
import os


class DrawCompound(object):
    def __init__(self, file_path_obj,scoreInstitution_obj):
        self.file_path = file_path_obj
        self.id_score_dict = scoreInstitution_obj.id_score_dict
        self.result = {}
        self.product_results()

    def product_results(self):
        items_iterator = iter(self.id_score_dict.items())
        first_50_items = list(islice(items_iterator, 20)) if len(self.id_score_dict) > 50 else list(self.id_score_dict.items())
        key_list = [item[0] for item in first_50_items]
        mol_dict = self.get_mol(key_list)
        rank = 0
        for key,value in first_50_items:
            rank += 1
            compound_score = value[0]
            green_index,green_db_data,green_user_data = [],[],[]
            yellow_index,yellow_db_data,yellow_user_data = [],[],[]
            orange_index,orange_db_data,orange_user_data = [],[],[]
            red_index,red_db_data,red_user_data = [],[],[]
            for green_data in value[1]:
                if green_data != 0 :
                    green_index.append(green_data[0])
                    green_db_data.append(green_data[1])
                    green_user_data.append(green_data[2])
            for yellow_data in value[2]:
                if yellow_data != 0 :
                    yellow_index.append(yellow_data[0])
                    yellow_db_data.append(yellow_data[1])
                    yellow_user_data.append(yellow_data[2])
            for orange_data in value[3]:
                if orange_data != 0 :
                    orange_index.append(orange_data[0])
                    orange_db_data.append(orange_data[1])
                    orange_user_data.append(orange_data[2])
            for red_data in value[4]:
                if red_data != 0 :
                    red_index.append(red_data[0])
                    red_db_data.append(red_data[1])
                    red_user_data.append(red_data[2])
            all_index = [i + 1  for i in green_index+yellow_index+orange_index+red_index]
            result_dataframe = pd.DataFrame({'atom_Index':all_index,
                                             'DB_data':green_db_data+yellow_db_data+orange_db_data+red_db_data,
                                             'User_data':green_user_data+yellow_user_data+orange_user_data+red_user_data})
            result_dataframe = result_dataframe.set_index('atom_Index')
            styled_df = result_dataframe.style.apply(self.row_color, axis=1,
                                       green_index=[i+1 for i in green_index],
                                       yellow_index=[i+1 for i in yellow_index],
                                       orange_index=[i+1 for i in orange_index],
                                       red_index=[i+1 for i in red_index])
            html = styled_df.to_html()
            ppm_img1=self.generate_ppm(green_db_data,yellow_db_data,orange_db_data,red_db_data)
            ppm_img2=self.generate_ppm(green_user_data,yellow_user_data,orange_user_data,red_user_data)


            MW = None
            fomular = None
            smiles = None
            compound_structure1, compound_structure2,compound_structure3 = None,None,None
            for k,v in mol_dict.items():
                if k == key:
                    mol = v[0]
                    AllChem.Compute2DCoords(mol)
                    try:
                      index_data = self.get_data_type(green_index,yellow_index,orange_index,red_index)
                      db_data = self.get_data_type(green_db_data,yellow_db_data,orange_db_data,red_db_data)
                      user_data = self.get_data_type(green_user_data,yellow_user_data,orange_user_data,red_user_data)
                      compound_structure1 = self.draw_compound(green_index,yellow_index,orange_index,red_index,mol,rank,(0,1,0),(1,1,0),(1,0.5,0),(1,0,0),index_data,1,0.7,1)
                      compound_structure2 = self.draw_compound(green_index,yellow_index,orange_index,red_index,mol,rank,(0,1,0),(1,1,0),(1,0.5,0),(1,0,0),db_data,2,0.6,0)
                      compound_structure3 = self.draw_compound(green_index,yellow_index,orange_index,red_index,mol,rank,(0,1,0),(1,1,0),(1,0.5,0),(1,0,0),user_data,3,0.6,0)

                    except:
                       compound_structure1 = None
                       compound_structure2 = None
                       compound_structure3 = None

                    MW = v[1]
                    fomular = v[2]
                    smiles = v[3]
            self.result[key] = [compound_score, rank,MW,html,compound_structure1,compound_structure2,compound_structure3,fomular,smiles,ppm_img1,ppm_img2]
        return self.result

    def row_color(self,row,green_index,yellow_index,orange_index,red_index):
        if row.name in green_index:
            return ['background-color: #00FF00'] * 2
        elif row.name in yellow_index:
            return ['background-color: #FFFF00'] * 2
        elif row.name in orange_index:
            return ['background-color: #FFA500'] * 2
        elif row.name in red_index:
            return ['background-color: #FF0000'] * 2
        else:
            return ['background-color: #FFFFFF'] * 2


    def draw_compound(self, green_index, yellow_index, orange_index, red_index, mol, rank, color1, color2, color3, color4, data_type, num, labelsize, index_num):
        # 确保IMG文件夹存在
        img_dir = "RESULT_IMG"
        os.makedirs(img_dir, exist_ok=True)  # 自动创建目录（如果不存在）

        drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
        dr = rdMolDraw2D.MolDrawOptions()
        dr.setBackgroundColour((173/255, 216/255, 230/255))
        dr.addAtomIndices = False
        dr.bondLineWidth = 2.5
        dr.dotsPerAngstrom = 1000
        for index, i in enumerate(green_index + yellow_index + orange_index + red_index):
            dr.atomLabels[i] = str(data_type[index] + index_num)
        dr.fixedBondLength = 70
        dr.baseFontSize = labelsize
        drawer.SetDrawOptions(dr)

        highlightAtomColors = {}
        highlightAtomColors.update({i: color1 for i in green_index})
        highlightAtomColors.update({i: color2 for i in yellow_index})
        highlightAtomColors.update({i: color3 for i in orange_index})
        highlightAtomColors.update({i: color4 for i in red_index})

        drawer.DrawMolecule(mol,
                            highlightAtoms=green_index + yellow_index + orange_index + red_index,
                            highlightAtomColors=highlightAtomColors)
        drawer.FinishDrawing()

        output_file = os.path.join(img_dir, f'rank_{rank}-{num}_molecule.png')
        png_data = drawer.GetDrawingText()

        # 保存到指定路径
        with open(output_file, 'wb') as f:
            f.write(png_data)

        return png_data

    def get_data_type(self,data1,data2,data3,data4):
        data_list = [int(i) for i in data1 + data2 + data3 + data4 if i != '']
        return data_list
    def get_mol(self, compound_id_list):
        filepaths = self.file_path
        mol_dict = {}
        for file_path in filepaths:
            supplier = Chem.SDMolSupplier(file_path)
            for mol in supplier:
              try:
                mol_id = mol.GetProp('ID')
                if mol_id in compound_id_list:
                    MW = mol.GetProp('FW')
                    formula = rdMolDescriptors.CalcMolFormula(mol)
                    smiles = str(Chem.MolToSmiles(mol))
                    mol_dict[mol_id] = [mol, MW,formula,smiles]
              except:
                  print('no valid mol found')
        return mol_dict


    def generate_ppm(self,greenlist,yellowlist,orangelist,red_list):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_xlim(220, 0)
        ax.set_ylim(0, 12000)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        ax.yaxis.set_minor_locator(plt.NullLocator())

        for strength in greenlist:
            ax.axvline(x=strength, ymax=0.8,color='green', linewidth=1)
        for strength in yellowlist:
            ax.axvline(x=strength, ymax=0.6,color='yellow', linewidth=1)
        for strength in orangelist:
            ax.axvline(x=strength, ymax=0.4,color='orange', linewidth=1)
        for strength in red_list:
            ax.axvline(x=strength, ymax=0.2,color='red', linewidth=1)


        ax.set_xlabel('ppm')
        ax.set_ylabel('strength')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        buffer = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buffer, format='png')
        binary_data = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        return  binary_data











#
#
#
# file_paths = File_path()
# user_path_init = Add_user_path()
# user_path = user_path_init.select_user_path()
# print(user_path)
# file_paths.select_user_path(user_path)
# # 选择模型路径、用户路径和实验数据路径
# # file_paths.select_model_path()

# all_paths = file_paths.select_all_paths()
# #rd = ProcesInputdata(
#     # '164,103,181,157,99,164,94,161,104,120,113,146,150,116,122,130,170,77,33,55,22,11,34,90,49,57,204,123,127,102,59,78,140,152,157',
#     # '104,157,146,122,161,113', '55,33', '104,157,146,122,161,113,78,140,152,157')
      #52,35,24,46,202,151,125,46,18,177,29,32,51,25,143,119,29,52,150,113,23,40,23,31,37,178,138,121,40,45,18
# rd = ProcesInputdata( '204,61,45,67,89,99,150.97, 140.11, 114,125.20, 120.45, 119.57, 118.45, 118.22, 111.58, 85.83, 84.79, 72.13, 53.09, 48.84, 46.50, 40.08, 37.74, 36.64, 33.92, 27.61, 26.18, 25.38, 24.73, 23.81, 22.06, 22.04, 22.044,20.06, 14.66, 12.77',' 120.45,61, 119.57, 118.45, 111.59, 85.84, 84.79, 48.85, 46.51, 26.18, 23.82, 22.06,22.044,20.07, 14.66, 45,67,89,12.77',' 60.53, 37.74, 33.93,114, 27.61, 25.38, 24.73,22.06, 22.04',' 120.45, 119.57, 118.46, 111.59, 85.84, 84.79, 48.85, 46.51, 37.75, 33.93, 27.61, 25.38, 24.73, 22.044')
# ctype = rd.match_type_state
