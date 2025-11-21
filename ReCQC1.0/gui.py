import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from SDF import File_path, Process_sdf_files,Add_user_path
from USERINPUT import ProcesInputdata
from MATCHINSTITUTE import ProductBestCompare,ScoreInstitution
from DRAWCOMPOUND import DrawCompound
from png import ViewPage,ResultViewerApp
from DBORGANIZATION import DBOrganization,Add_file_path
from similarsearch import SimilarSearch, GetSubstructures

class Main():

    def __init__(self,master:tk.Tk):
        self.root = master
        self.root.title("ReCQC 1.0")
        self.root.geometry("800x600")
        self.icon_path = 'tcalxy.jpg'
        self.icon = self.get_icon()
        self.root.iconphoto(True, self.icon)
        self.creat_main_page()



    def get_icon(self):
        icon = Image.open(self.icon_path)
        icon1 = icon.resize((50, 50),  Image.LANCZOS)
        icon_photo = ImageTk.PhotoImage(icon1)
        return icon_photo

    def creat_main_page(self):
        self.primary_page_frame = PrimaryPage(self.root)
        self.db_page_frame = DBPage(self.root)
        self.query_page_frame = QueryPage(self.root)
        self.about_page_frame = AboutPage(self.root)
        self.primary_page_frame.pack(fill="both", expand=True)
        menu = tk.Menu(self.root)
        menu.add_command(label="Main", command=self.show_main_page)
        menu.add_command(label="DatabaseTool", command=self.creat_db_page)
        menu.add_command(label="About", command=self.about_us)
        self.root.config(menu=menu, bg="grey",bd=1)
        self.root.mainloop()


    def show_main_page(self):
        self.primary_page_frame.pack(fill="both", expand=True)
        self.db_page_frame.pack_forget()
        self.query_page_frame.pack_forget()
        self.about_page_frame.pack_forget()


    def creat_db_page(self):
        self.primary_page_frame.pack_forget()
        self.db_page_frame.pack(fill = "both", expand=True)
        self.query_page_frame.pack_forget()
        self.about_page_frame.pack_forget()


    def about_us(self):
        self.primary_page_frame.forget()
        self.db_page_frame.pack_forget()
        self.query_page_frame.pack_forget()
        self.about_page_frame.pack(fill = "both", expand=True)



class PrimaryPage(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.configure(bg='#2C3E50')

        self.parameter_frame = ParameterPage(root, self.goto_primary_page,filepath_obj=File_path(),usepath_obj=Add_user_path())# 创建搜索页面框架，但初始时隐藏
        self.place(relwidth=1, relheight=1)  # 使PrimaryPage填充整个root窗口
        self.create_primary_page()

    def create_primary_page(self):
        self.pure_spectrum_query_button = tk.Button(self, text="Spectrum Search", font=('Arial', '20','bold'), bg='grey',
                                                    command=self.goto_parameter_page)
        self.pure_spectrum_query_button.place(relx=0.5, rely=0.7, anchor='center')

        self.main_page_label1 = tk.Label(self, text="Natural Compounds Analyse", font=('Arial', '25', 'italic', 'bold'),
                                         bg='#2C3E50', fg='grey')
        self.main_page_label1.place(relx=0.5, rely=0.3, anchor='center')
        self.main_page_label2 = tk.Label(self, text="ReCQC-1.0", font=('Arial', '25', 'italic', 'bold'),
                                         bg='#2C3E50', fg='grey')
        self.main_page_label2.place(relx=0.5, rely=0.4, anchor='center')
        self.main_page_label3 = tk.Label(self, text="a software designed for analyse mixture or pure compounds",
                                         font=('Arial', '15', 'italic', 'bold'), bg='#2C3E50', fg='black')
        self.main_page_label3.place(relx=0.5, rely=0.5, anchor='center')


    def goto_parameter_page(self):
        self.place_forget()  # 隐藏当前页面
        self.parameter_frame.place(relwidth=1, relheight=1)  # 显示搜索页面


    def goto_primary_page(self):
        self.parameter_frame.place_forget()
        self.place(relwidth=1, relheight=1)

    def place_forget(self):

        self.place(relwidth=0, relheight=0)


class ParameterPage(tk.Frame):
    def __init__(self, root, callback, filepath_obj,usepath_obj):
        super().__init__(root, bg='#2C3E50')
        self.callback = callback
        self.usepath_obj = usepath_obj
        self.filepath_obj = filepath_obj
        self.usepath = []
        self.create_parameter_page()

    # def go_to_search(self):
    #     green_threshold, yellow_threshold, min_mw, max_mw, min_c_num, max_c_num, skeleton_threshold = self.get_green_yellow_minc_maxc_minmw_maxmw_skeleton()
    #     self.search_frame = SearchPage(root, self.goto_primary_page,self.filepath_obj,self.usepath,green_threshold, yellow_threshold,skeleton_threshold, min_c_num, max_c_num ,min_mw, max_mw,)  # 创建搜索页面框架，但初始时隐藏
    #
    #

    def create_parameter_page(self):

        self.back_button = tk.Button(self, text="Back", font=('Arial', '12'), bg='grey', command=self.callback)
        self.back_button.place(relx=0.2, rely=0.8, anchor='center')
        self.search_button = tk.Button(self, text="Next", font=('Arial', '12'), bg='grey', command=self.goto_search_page)
        self.search_button.place(relx=0.8, rely=0.8, anchor='center')
        # 创建“库选择”标签
        self.library_label = tk.Label(self, text="Library Selection", font=('Arial', '12'), bg='#FFFFFF', fg='black',
                                      borderwidth=2, relief='solid', padx=5, pady=2)
        self.library_label.config(bg='black', fg='white')
        self.library_label.place(relx=0.1, rely=0.1, anchor='w')
        self.library_label = tk.Label(self, text="Parameter Settings", font=('Arial', '12'), bg='#FFFFFF', fg='black',
                                      borderwidth=2, relief='solid', padx=5, pady=2)
        self.library_label.config(bg='black', fg='white')
        self.library_label.place(relx=0.1, rely=0.3, anchor='w')

        self.checkbox_var1 = tk.IntVar()
        self.checkbox_var2 = tk.IntVar()
        self.checkbox_var3 = tk.IntVar()

        self.checkbox1 = tk.Checkbutton(self, text="Experimental DB", variable=self.checkbox_var1, font=('Arial', '12'))
        self.checkbox2 = tk.Checkbutton(self, text="Simulated DB", variable=self.checkbox_var2, font=('Arial', '12'))
        self.checkbox3 = tk.Checkbutton(self, text="Predicted DB", variable=self.checkbox_var3, font=('Arial', '12'))

        self.checkbox1.place(relx=0.3, rely=0.1, anchor='w')
        self.checkbox2.place(relx=0.5, rely=0.1, anchor='w')
        self.checkbox3.place(relx=0.7, rely=0.1, anchor='w')
        self.checkbox_var1.trace("w", lambda name, index, mode, *args: self.select_expriment_library() )
        self.checkbox_var2.trace("w",lambda name, index, mode, *args: self.select_model_library())
        self.select_custom_library_button = tk.Button(self, text=" Select Custom Library", font=('Arial', '12'), bg='#FFFFFF', fg='black',
                                                      borderwidth=2, relief='solid', command=self.select_custom_library)
        self.select_custom_library_button.place(relx=0.3, rely=0.20, anchor='w')

        self.select_custom_library_button = tk.Button(self, text="Remove Custom Library", font=('Arial', '12'), bg='#FFFFFF', fg='black',
                                                      borderwidth=2, relief='solid', command=self.delete_custom_library)
        self.select_custom_library_button.place(relx=0.6, rely=0.20, anchor='w')

        self.first_toler_lalue = tk.Label(self, text="First Tolerance", font=('Arial', '12'), bg='black', fg='#FFFFFF',
                                      borderwidth=2, relief='solid', padx=5, pady=2)
        self.first_toler_lalue.config(bg='grey', fg='black')
        self.first_toler_lalue.place(relx=0.3, rely=0.3, anchor='w')
        self.first_toler_default = tk.StringVar(value='2.0')
        self.first_toler_lalue_entry = tk.Entry(self, font=('Arial', '10'), bd=2, relief='solid', bg='white', insertwidth=2, textvariable=self.first_toler_default)
        self.first_toler_lalue_entry.place(relx=0.5, rely=0.3, anchor='w', width=100)
        self.second_toler_lalue = tk.Label(self, text="Second Tolerance", font=('Arial', '12'), bg='black', fg='#FFFFFF',
                                      borderwidth=2, relief='solid', padx=5, pady=2)
        self.second_toler_lalue.config(bg='grey', fg='black')
        self.second_toler_lalue.place(relx=0.3, rely=0.4, anchor='w')
        self.second_toler_default = tk.StringVar(value='4.0')
        self.second_toler_entry = tk.Entry(self, font=('Arial', '10'), bd=2, relief='solid', bg='white', insertwidth=2, textvariable=self.second_toler_default)
        self.second_toler_entry.place(relx=0.5, rely=0.4, anchor='w', width=100)

        self.mw_lalue = tk.Label(self, text="Mass Range", font=('Arial', '12'), bg='black', fg='#FFFFFF',
                                      borderwidth=2, relief='solid', padx=5, pady=2)
        self.mw_lalue.config(bg='grey', fg='black')
        self.mw_lalue.place(relx=0.3, rely=0.5, anchor='w')
        self.min_mw_num_default = tk.StringVar(value='0')
        self.min_entry = tk.Entry(self, font=('Arial', '10'), bd=2, relief='solid', bg='white', insertwidth=2, textvariable=self.min_mw_num_default)
        self.min_entry.place(relx=0.5, rely=0.5, anchor='w', width=65)
        self.to = tk.Label(self, text="——", font=('Arial', '12'),bg='#2C3E50')
        self.to = self.to.place(relx=0.6, rely=0.5, anchor='w')
        self.max_mw_num_default = tk.StringVar(value='1000')
        self.max_entry = tk.Entry(self, font=('Arial', '10'), bd=2, relief='solid', bg='white', insertwidth=2, textvariable=self.max_mw_num_default)
        self.max_entry.place(relx=0.68, rely=0.5, anchor='w', width=65)

        self.c_num_label = tk.Label(self, text="Carbon Number", font=('Arial', '12'), bg='black', fg='#FFFFFF', borderwidth=2, relief='solid', padx=5, pady=2)
        self.c_num_label.config(bg='grey', fg='black')
        self.c_num_label.place(relx=0.3, rely=0.6, anchor='w')
        self.default_min_c_text = tk.StringVar(value="0")
        self.min_c_num_entry = tk.Entry(self, font=('Arial', '10'), bd=2, relief='solid', bg='white', insertwidth=2,textvariable=self.default_min_c_text)
        self.min_c_num_entry.place(relx=0.5, rely=0.6, anchor='w', width=65)
        self.to2 = tk.Label(self, text="——", font=('Arial', '16'), bg='#2C3E50')
        self.to2.place(relx=0.6, rely=0.6, anchor='w')
        self.default_max_c_text = tk.StringVar(value="100")
        self.max_c_num_entry = tk.Entry(self, font=('Arial', '10'), bd=2, relief='solid', bg='white', insertwidth=2,textvariable=self.default_max_c_text)
        self.max_c_num_entry.place(relx=0.68, rely=0.6, anchor='w', width=65)

        self.skeleton_label = tk.Label(self, text="Skeleton Similarity", font=('Arial', '12'), bg='black', fg='#FFFFFF', borderwidth=2, relief='solid', padx=5, pady=2)
        self.skeleton_label.config(bg='grey', fg='black')
        self.skeleton_label.place(relx=0.3, rely=0.7, anchor='w')
        self.default_skeleton_text = tk.StringVar(value="0.85")
        self.skeleton_entry = tk.Entry(self, font=('Arial', '10'), bd=2, relief='solid', bg='white', insertwidth=2,textvariable=self.default_skeleton_text)
        self.skeleton_entry.place(relx=0.5, rely=0.7, anchor='w', width=65)

    def get_green_yellow_minc_maxc_minmw_maxmw_skeleton(self):
        green_threshold = float(self.first_toler_default.get())
        yellow_threshold = float(self.second_toler_default.get())
        min_c_num = int(self.default_min_c_text.get())
        max_c_num = int(self.default_max_c_text.get())
        min_mw = float(self.min_mw_num_default.get())
        max_mw = float(self.max_mw_num_default.get())
        skeleton_threshold = float(self.default_skeleton_text.get())
        return green_threshold, yellow_threshold,  min_mw, max_mw, min_c_num, max_c_num,skeleton_threshold

    def select_custom_library(self):
        self.usepath = self.usepath_obj.select_user_path()
        return self.usepath

    def delete_custom_library(self):
        self.usepath = []
        return self.usepath
    def select_expriment_library(self,event=None):
        if self.checkbox_var1.get():
            self.filepath_obj.select_realdata_path()
        else:
            self.filepath_obj.remove_realdata_path()

    def select_model_library(self,event=None):
        if self.checkbox_var2.get():
            self.filepath_obj.select_model_path()
        else:
            self.filepath_obj.remove_model_path()

    def select_prediction_library(self):
        if self.checkbox_var3.get():
            self.filepath_obj.select_prediction_path()
        else:
            self.filepath_obj.remove_prediction_path()

    def goto_search_page(self):
        self.place_forget()
        green_threshold, yellow_threshold, min_mw, max_mw, min_c_num, max_c_num, skeleton_threshold = self.get_green_yellow_minc_maxc_minmw_maxmw_skeleton()
        self.search_frame = SearchPage(root, self.goto_primary_page,self.filepath_obj,self.usepath,green_threshold, yellow_threshold,skeleton_threshold, min_c_num, max_c_num ,min_mw, max_mw,)  # 创建搜索页面框架，但初始时隐藏


        self.search_frame.place(relwidth=1, relheight=1)  # 显示搜索页面

    def goto_primary_page(self):
        self.search_frame.place_forget()  # 隐藏搜索页面
        self.place(relwidth=1, relheight=1)  # 显示当前页面

    def place_forget(self):

        self.place(relwidth=0, relheight=0)  #


class DBPage(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.configure(bg='#2C3E50')
        self.folder_path = ''
        self.creat_db_page()
    def creat_db_page(self):
        generate_label = tk.Label(self,text='generate db',font=('Arial', '15'), bg='black', fg='white')
        generate_label.place(relx=0.4, rely=0.22, anchor='w', width=200)
        select_folder_button = tk.Button(self, text="select a folder", font=('Arial', '13'), bg='#FFFFFF', fg='black',
                                                    relief='solid',command=self.select_folder_path)
        select_folder_button.config(bg='grey', fg='black')
        select_folder_button.place(relx=0.2, rely=0.05, anchor='w', width=200)
        delete_folder_button = tk.Button(self,text='delete a folder', font=('Arial', '13'), bg='#FFFFFF', fg='black',
                                         relief='solid',command=self.delete_folder_path)
        delete_folder_button.config(bg='grey', fg='black')
        delete_folder_button.place(relx=0.55, rely=0.05, anchor='w', width=200)
        label_name = tk.Label(self,text='selected path :',font=('Arial', '10'), bg='#2C3E50', fg='black')
        label_name.place(relx=0.2, rely=0.15, anchor='w', width=200)
        self.file_label = tk.Text(self,font=('Arial', '10'), bg='#2C3E50', fg='black',width=400,height=1)
        self.file_label.insert(tk.END,'None')
        self.file_label.place(relx=0.55, rely=0.15, anchor='w', width=200)
        generate_button = tk.Button(self,text='Generate DataBase',font=('Arial', '10'), bg='#FFFFFF', fg='black',command=self.generate_sdf)
        generate_button.config(bg='grey', fg='black')
        generate_button.place(relx=0.2, rely=0.3, anchor='w', width=200)
        check_button = tk.Button(self,text='Check Folder',font=('Arial', '10'), bg='#FFFFFF', fg='black',command=self.check_folder)
        check_button.config(bg='grey', fg='black')
        check_button.place(relx=0.55, rely=0.3, anchor='w', width=200)

        similar_label = tk.Label(self,text='similar_search',font=('Arial', '15'), bg='black', fg='white')
        similar_label.place(relx=0.4, rely=0.4, anchor='w', width=200)
        similar_parameter_label = tk.Label(self,text='similar_threshold :',font=('Arial', '10'), bg='#2C3E50', fg='white')
        similar_parameter_label.place(relx=0.5, rely=0.5, anchor='w', width=200)

        self.similar_value = tk.StringVar(value='0.8')
        similar_parameter_entry = tk.Entry(self, font=('Arial', '10'), bd=2, relief='solid', bg='white', insertwidth=2, textvariable=self.similar_value)
        similar_parameter_entry.place(relx=0.75, rely=0.5, anchor='w', width=40)

        similar_radius_parameter_label = tk.Label(self,text='radius_value :',font=('Arial', '10'), bg='#2C3E50', fg='white')
        similar_radius_parameter_label.place(relx=0.05, rely=0.5, anchor='w', width=200)
        self.radius_value = tk.StringVar(value='3')
        similar_parameter_entry = tk.Entry(self, font=('Arial', '10'), bd=2, relief='solid', bg='white', insertwidth=2, textvariable=self.radius_value)
        similar_parameter_entry.place(relx=0.3, rely=0.5, anchor='w', width=40)

        smiles_label = tk.Label(self,text='target_smiles :',font=('Arial', '10'), bg='#2C3E50', fg='white')
        smiles_label.place(relx=0.05, rely=0.6, anchor='w', width=200)
        self.smiles_str = tk.StringVar(value='None')
        smiles_entry = tk.Entry(self, font=('Arial', '10'), bd=2, relief='solid', bg='white', insertwidth=2, textvariable=self.smiles_str)
        smiles_entry.place(relx=0.3, rely=0.6, anchor='w', width=200)

        similar_search_button = tk.Button(self,text='Search',font=('Arial', '13'), bg='#FFFFFF', fg='black',command=self.similar_search)
        similar_search_button.config(bg='grey', fg='black')
        similar_search_button.place(relx=0.7, rely=0.6, anchor='w', width=80)

        substructure_search = tk.Label(self,text='substructure_search',font=('Arial', '15'), bg='black', fg='white')
        substructure_search.place(relx=0.4, rely=0.7, anchor='w', width=200)
        sub_smiles_label = tk.Label(self,text='sub_smiles :',font=('Arial', '10'), bg='#2C3E50', fg='white')
        sub_smiles_label.place(relx=0.05, rely=0.8, anchor='w', width=200)
        self.sub_smiles = tk.StringVar(value='None')
        sub_smiles_entry = tk.Entry(self, font=('Arial', '10'), bd=2, relief='solid', bg='white', insertwidth=2, textvariable=self.sub_smiles)
        sub_smiles_entry.place(relx=0.3, rely=0.8, anchor='w', width=200)
        sub_search_button = tk.Button(self,text='Search',font=('Arial', '13'), bg='grey', fg='black',command=self.get_substructure_search)
        sub_search_button.config(bg='grey', fg='black')
        sub_search_button.place(relx=0.7, rely=0.8, anchor='w', width=80)


    def select_folder_path(self):
        self.folder_path_obj = Add_file_path()
        self.folder_path = self.folder_path_obj.select_user_path()
        self.update_label()

    def delete_folder_path(self):
        self.folder_path = ''
        self.update_label()

    def update_label(self):
        self.file_label.delete(1.0, tk.END)  # 删除现有内容
        self.file_label.insert(tk.END, self.folder_path)

    def check_folder(self):
        if self.folder_path == '':
            print('No folder selected')
        else:
            self.check_folder_obj = DBOrganization(self.folder_path)
            self.check_folder_obj.check_sdf_excel()
            self.check_folder_obj.check_sdf_mol()
            print('completed')

    def generate_sdf(self):
        if self.folder_path == '':
           print('No folder selected')
        else:
           self.check_folder_obj = DBOrganization(self.folder_path)
           self.check_folder_obj.write_sdf()
           print('completed')

    def similar_search(self):
        radius=3
        try:
            radius = int(self.radius_value.get())
        except ValueError:
            print('radius value must be an integer')
        smiles = self.smiles_str.get()
        threshold = float(self.similar_value.get())
        similarsearch_obj = SimilarSearch(smiles,self.folder_path,threshold, radius)
        similarsearch_obj.get_similars()

    def get_substructure_search(self):
        smiles = self.sub_smiles.get()
        sub_search_obj = GetSubstructures(smiles,self.folder_path)
        sub_search_obj.get_substructures_sdf()

class SearchPage(tk.Frame):
    def __init__(self, root,callback,filepath_obj,usepath, green_threshold_obj,yellow_threshold_obj,
                 skeleton_threshold_obj,min_c_num_obj,max_c_num_obj,min_mw_obj,max_mw_obj):
        super().__init__(root)
        self.configure(bg='#2C3E50')
        self.callback = callback
        self.filepath_obj = filepath_obj
        self.usepath_obj = usepath
        self.green_threshold_obj = green_threshold_obj
        self.yellow_threshold_obj = yellow_threshold_obj
        self.skeleton_threshold_obj = skeleton_threshold_obj
        self.max_c_num_obj = max_c_num_obj
        self.min_c_num_obj = min_c_num_obj
        self.max_mw_obj = max_mw_obj
        self.min_mw_obj = min_mw_obj
        self.creat_search_page()
    def creat_search_page(self):
        self.back_button = tk.Button(self, text="Back", font=('Arial', '16'), bg='grey', command=self.callback)
        self.back_button.place(relx=0.2, rely=0.85, anchor='center')
        self.search_button = tk.Button(self, text="Search", font=('Arial', '16'), bg='grey', command=self.search)
        self.search_button.place(relx=0.8, rely=0.85, anchor='center')

        self.all_c_label = tk.Label(self,text = 'All_C',font=('Arial', '13'), bg='black', fg='#FFFFFF', borderwidth=2, relief='solid', padx=5, pady=2)
        self.all_c_label.place(relx=0.1, rely=0.1, anchor='center')
        self.all_c_label_box = tk.Text(self, font=('Arial', '10'), bd=2, relief='solid', bg='#CCCCCC', insertwidth=2,width=70,height=5)
        self.all_c_label_box.place(relx=0.5, rely=0.1, anchor='center')

        self.up_dept135_label = tk.Label(self,text = 'Up_DEPT135',font=('Arial', '10'), bg='black', fg='white', borderwidth=2, relief='solid', padx=5, pady=2)
        self.up_dept135_label.place(relx=0.1, rely=0.3, anchor='center')
        self.up_dept135_label_box = tk.Text(self, font=('Arial', '10'), bd=2, relief='solid', bg='#CCCCCC', insertwidth=2,width=70,height=5)
        self.up_dept135_label_box.place(relx=0.5, rely=0.3, anchor='center')
        self.down_dept135_label = tk.Label(self,text = 'Down_DEPT135',font=('Arial', '10'), bg='black', fg='#FFFFFF', borderwidth=2, relief='solid', padx=5, pady=2)
        self.down_dept135_label.place(relx=0.1, rely=0.5, anchor='center')
        self.down_dept135_label_box = tk.Text(self, font=('Arial', '10'), bd=2, relief='solid', bg='#CCCCCC', insertwidth=2,width=70,height=5)
        self.down_dept135_label_box.place(relx=0.5, rely=0.5, anchor='center')
        self.dept90_label = tk.Label(self,text = 'DEPT90',font=('Arial', '10'), bg='black', fg='#FFFFFF', borderwidth=2, relief='solid', padx=5, pady=2)
        self.dept90_label.place(relx=0.1, rely=0.7, anchor='center')
        self.dept90_label_box = tk.Text(self, font=('Arial', '10'), bd=2, relief='solid', bg='#CCCCCC', insertwidth=2,width=70,height=5)
        self.dept90_label_box.place(relx=0.5, rely=0.7, anchor='center')

    def search(self):
        allc = self.all_c_label_box.get(1.0, 'end-1c')
        up_dept135 = self.up_dept135_label_box.get(1.0, 'end-1c')
        down_dept135 = self.down_dept135_label_box.get(1.0, 'end-1c')
        dept90 = self.dept90_label_box.get(1.0, 'end-1c')
        self.filepath_obj.select_user_path(self.usepath_obj)
        all_paths = self.filepath_obj.select_all_paths()

        input_data = ProcesInputdata(allc, up_dept135, down_dept135, dept90)
        c_type = input_data.match_type_state


        sdf_result = Process_sdf_files(all_paths, c_type, self.min_mw_obj, self.max_mw_obj, self.min_c_num_obj, self.max_c_num_obj)
        compare_data = ProductBestCompare(input_data, sdf_result,self.green_threshold_obj)
        #print(compare_data.all_match_data_dict)
        score = ScoreInstitution(compare_data,self.yellow_threshold_obj, self.skeleton_threshold_obj)
        img_result = DrawCompound(all_paths,score)
        result_dict = img_result.result
        result_page = ResultViewerApp(root,result_dict,callback=lambda idx: print(f"Global index clicked: {idx}"))
        result_page.mainloop()

class QueryPage(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.configure(bg='#2C3E50')
        self.creat_help_page()

    def creat_help_page(self):
        pass


class AboutPage(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.configure(bg='#2C3E50')
        self.creat_about_page()

    def creat_about_page(self):
        self.label1 = tk.Label(self, bg='#2C3E50',text='Team：Natural Products Research and \n Application Team of Yunnan University',font=('Arial',15),fg='black')
        self.label1.place(relx=0.5, rely=0.15, anchor='center')
        self.label2 = tk.Label(self,bg='#2C3E50',text='Version：ReCQC-1.0.0',font=('Arial',15),fg='black')
        self.label2.place(relx=0.5, rely=0.25, anchor='center')
        self.label3 = tk.Label(self,bg='#2C3E50',text='Email：2638614209@qq.com',font=('Arial',15),fg='black')
        self.label3.place(relx=0.5, rely=0.35, anchor='center')


if __name__ == '__main__':
    root = tk.Tk()
    Main(root)

