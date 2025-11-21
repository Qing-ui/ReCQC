import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io
from tkinter import messagebox
import webbrowser


class ViewPage(tk.Frame):
    def __init__(self, root,callback, search_results):
        super().__init__(root)
        self.configure(bg='#2C3E50')
        self.callback = callback
        self.search_results = search_results[:6] # 每次只显示前6个结果
        self.creat_result_page()

    def creat_result_page(self):
        for idx,item in enumerate(self.search_results):
            item_frame = ttk.Frame(self)
            item_frame.grid(row=idx//3, column=idx%3, padx=25, pady=15, sticky='nsew')
            tk.Grid.rowconfigure(self, idx//3, weight=1)
            tk.Grid.columnconfigure(self, idx%3, weight=1)
            img_data = item[1][4]
            img = Image.open(io.BytesIO(img_data))
            img = img.resize((200,200),Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            # 创建图片控件
            # 创建图片控件并放入Frame中
            img_label = tk.Label(item_frame, image=photo, compound='center')
            img_label.image = photo  # 保持对图片的引用
            img_label.pack(fill=tk.BOTH, expand=True)

            btn = ttk.Button(item_frame, text='Detail',command=lambda idx=idx :self.callback(idx))
            btn.pack( padx=5, pady=5, side=tk.RIGHT)
            label = ttk.Label(item_frame, text='Score：'+str(item[1][0]), justify='left')
            label.pack(padx=5, pady=5, side=tk.LEFT)


        for r in range(len(self.search_results), 6):
            for c in range(3):
                tk.Grid.rowconfigure(self, r//3, minsize=0)
                tk.Grid.columnconfigure(self, c, minsize=0)


class ResultViewerApp(tk.Toplevel):
    def __init__(self, root,search_results,callback):
        super().__init__(root)
        self.geometry("800x650")
        self.configure(bg='#2C3E50')
        self.title("Result Viewer")
        self.callback = callback
        self.search_results = list(search_results.items())
        self.current_index = 0
        self.pages = [self.search_results[i:i+6] for i in range(0, len(self.search_results), 6)]
        self.create_widgets()

    def create_widgets(self):
        self.frame = tk.Frame(self)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # 初始化第一页
        self.show_page(0)

        # 创建下一页和上一页的按钮
        prev_btn = ttk.Button(self, text="Previous", command=self.prev_page)
        prev_btn.pack(side=tk.LEFT, padx=10, pady=10)

        next_btn = ttk.Button(self, text="Next", command=self.next_page)
        next_btn.pack(side=tk.RIGHT, padx=10, pady=10)

    def show_page(self, index):
        try:
            for widget in self.frame.winfo_children():
                widget.destroy()  # 清除当前页面上的所有控件
            page_results = self.pages[index]
            search_page = ViewPage(self.frame, self.on_button_click, page_results)
            search_page.pack(fill=tk.BOTH, expand=True)
            self.current_index = index
        except IndexError:
            print('no page')

    def prev_page(self):
        if self.current_index > 0:
            self.show_page(self.current_index - 1)

    def next_page(self):
        if self.current_index < len(self.pages) - 1:
            self.show_page(self.current_index + 1)

    def on_button_click(self, idx):
        # 这里处理按钮点击事件，idx是当前点击的按钮对应的搜索结果索引（在当前页内的索引）
        global_idx = self.current_index * 6 + idx
        frame = tk.Frame(self, bg='#2C3E50')
        frame.pack(fill=tk.BOTH, expand=True)
        detial_page = DetailPage(frame,self.search_results[global_idx])
        print(f"Button clicked for result {global_idx}")


class DetailPage(tk.Toplevel):
    def __init__(self, root,item):
        super().__init__(root)
        self.geometry("1900x1000")
        self.configure(bg='#2C3E50')
        self.title("Detail Page")
        self.item = item
        self.create_detail_page()

    def create_detail_page(self):
        rank_label = tk.Label(self, text="Rank: "+f'{self.item[1][1]}', justify='left',font=("Arial", 13))
        rank_label.place(relx=0.05, rely=0.05)

        score_label = tk.Label(self,text='Score: '+f'{self.item[1][0]}', justify='left',font=("Arial", 13))
        score_label.place(relx=0.2, rely=0.05)

        id_label = tk.Label(self,text='ID: '+f'{self.item[0]}', justify='left',font=("Arial", 13))
        id_label.place(relx=0.4, rely=0.05)

        mw_label = tk.Label(self,text='MW: '+f'{self.item[1][2]}', justify='left',font=("Arial", 13))
        mw_label.place(relx=0.6, rely=0.05)

        fomular_label = tk.Label(self,text='fomular：'+f'{self.item[1][7]}', justify='left',font=("Arial", 13))
        fomular_label.place(relx=0.8, rely=0.05)

        smiles_label = tk.Label(self,text='smiles: ', justify='left',font=("Arial", 13))
        smiles_label.place(relx=0.05, rely=0.15)

        smiles_entry = tk.Text(self, width=100,font=("Arial", 13),fg='white',bg='#2C3E50',height='1')
        smiles_entry.insert(tk.END, self.item[1][8])  # 使用tk.END作为插入点
        smiles_entry.place(relx=0.2, rely=0.15)

        img_data1 = self.item[1][4]
        img_data2 = self.item[1][5]
        img_data3 = self.item[1][6]

        # 加载和缩放图像
        img1 = Image.open(io.BytesIO(img_data1)).resize((500, 500), Image.LANCZOS)
        img2 = Image.open(io.BytesIO(img_data2)).resize((500, 500), Image.LANCZOS)
        img3 = Image.open(io.BytesIO(img_data3)).resize((500, 500), Image.LANCZOS)

        # 转换为 PhotoImage 对象
        photo1 = ImageTk.PhotoImage(img1)
        photo2 = ImageTk.PhotoImage(img2)
        photo3 = ImageTk.PhotoImage(img3)

        # 存储图像和对应的 PhotoImage 对象
        self.images = {
            'IndexImage': photo1,
            'DataBaseImage': photo2,
            'UserImage': photo3
        }

        # 默认显示第一张图片
        self.current_image_key = 'IndexImage'
        self.image_label = tk.Label(self)
        self.image_label.config(image=self.images[self.current_image_key])
        self.image_label.image = self.images[self.current_image_key]
        self.image_label.place(relx=0.05, rely=0.3,)

        # 创建单选框
        self.radio_var = tk.StringVar(value=self.current_image_key)
        radio_frame = tk.Frame(self)
        radio_frame.place(relx=0.05, rely=0.25)  # 使用 pack 布局管理器，并添加一些内边距

        radio_cols = 0
        for key, _ in self.images.items():
            tk.Radiobutton(radio_frame, text=f'Show {key}', variable=self.radio_var, value=key,font=('Arial', 12),
                           command=lambda key=key: self.update_image(key)).grid(row=0, column=radio_cols)
            radio_cols += 1
        html = self.item[1][3]
        open_button = tk.Button(self, text="Open Df_html Table", command=lambda: self.open_html_table(html))
        open_button.place(relx=0.5, rely=0.25)
        html_label = tk.Label(self, text="(Here's a representation of a web table with labeled results)", justify='left',font=("Arial", 13),bg='#2C3E50',fg='white')
        html_label.place(relx=0.6, rely=0.25)

        ppm_data1 = self.item[1][9]
        ppm_data2 = self.item[1][10]

        # 加载和缩放图像
        ppm1 = Image.open(io.BytesIO(ppm_data1)).resize((650, 400), Image.LANCZOS)
        ppm2 = Image.open(io.BytesIO(ppm_data2)).resize((650, 400), Image.LANCZOS)

        photo4 = ImageTk.PhotoImage(ppm1)
        photo5 = ImageTk.PhotoImage(ppm2)

        # 存储图像和对应的 PhotoImage 对象
        self.ppm_images = {
            'db_ppm': photo4,
            'user_ppm': photo5
        }

        # 默认显示第一张图片
        self.current_ppm_key = 'db_ppm'
        self.ppm_label = tk.Label(self)
        self.ppm_label.config(image=self.ppm_images[self.current_ppm_key])
        self.ppm_label.image = self.ppm_images[self.current_ppm_key]  # 保持对图片的引用
        self.ppm_label.place(relx=0.5, rely=0.4,)# 或者使用其他布局管理器

        # 创建单选框
        self.ppm_radio_var = tk.StringVar(value=self.current_ppm_key)
        radio_ppm_frame = tk.Frame(self)
        radio_ppm_frame.place(relx=0.6, rely=0.35)  # 使用 pack 布局管理器，并添加一些内边距

        radio_ppm_cols = 0
        for key, _ in self.ppm_images.items():
            tk.Radiobutton(radio_ppm_frame, text=f'Show {key}', variable=self.ppm_radio_var, value=key,font=('Arial', 12),
                           command=lambda key=key: self.update_image2(key)).grid(row=0, column=radio_ppm_cols)
            radio_ppm_cols += 1



    def update_image(self, key):
        self.image_label.config(image=self.images[key])
        self.image_label.image = self.images[key]
        self.current_image_key = key

    def update_image2(self, key):
        self.ppm_label.config(image=self.ppm_images[key])
        self.ppm_label.image = self.ppm_images[key]
        self.current_ppm_key = key


    def open_html_table(self,df_html):
        html_content = df_html
        with io.BytesIO() as buffer:
            buffer.write(html_content.encode('utf-8'))
            buffer.seek(0)
            html_file_path = 'temp_table.html'
            with open(html_file_path, 'wb') as html_file:
                html_file.write(buffer.read())
        # 使用 webbrowser 打开 HTML 文件
        try:
            webbrowser.open(html_file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open HTML file: {e}")




