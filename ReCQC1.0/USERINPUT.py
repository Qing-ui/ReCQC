import itertools
class ReceiveData:
    def __init__(self,all_spectrum_str,up_dept135_str,down_dept135_str,dept90_str):
        self.all_spectrum_str = all_spectrum_str
        self.up_dept135_str = up_dept135_str
        self.down_dept135_str = down_dept135_str
        self.dept135_str = self.up_dept135_str + self.down_dept135_str
        self.dept90_str = dept90_str

        self.all_spectrum = []
        self.down_dept135 = []
        self.up_dept135 = []
        self.dept135 = self.up_dept135 + self.down_dept135
        self.dept90 = []
        self.check_acquire_spectrum_str()
        self.match_state = None
        self.back_state()
        self.match_type_state = None
        self.back_match_type_state()

    def check_acquire_spectrum_str(self):
        errors = []
        if self.all_spectrum_str == '' and self.dept135_str == '' and self.dept90_str == '':
            error = 'The input data must not be empty'
            errors.append(error)
        else:
            if self.all_spectrum_str =='':
                self.all_spectrum = []
            else:
                try:
                    spectrum_list = self.all_spectrum_str.split(',')
                    for spectrum in spectrum_list:
                        self.all_spectrum.append(float(spectrum))

                except:
                    error = 'The input data is in the wrong format'
                    errors.append(error)
            if self.dept135_str == '':
                self.dept135 = []
            else:
                try:
                    if self.up_dept135_str == '':
                        self.up_dept135 = []
                    else:
                        spectrum_list = self.up_dept135_str.split(',')
                        for spectrum in spectrum_list:
                            self.up_dept135.append(float(spectrum))
                        #print(self.up_dept135)


                    if self.down_dept135_str == '':
                        self.down_dept135 = []
                    else :
                        spectrum_list = self.down_dept135_str.split(',')
                        for spectrum in spectrum_list:
                            self.down_dept135.append(float(spectrum))

                    self.dept135 = self.up_dept135 + self.down_dept135
                    #print(self.down_dept135)

                except:
                    error = 'The input data is in the wrong format'
                    errors.append(error)
            if self.dept90_str == '':
                self.dept90 = []
            else:
                try:
                    spectrum_list = self.dept90_str.split(',')
                    for spectrum in spectrum_list:
                        self.dept90.append(float(spectrum))
                except:
                    error = 'The input data is in the wrong format'
                    errors.append(error)
        if len(errors) == 0:
            return self.all_spectrum,self.dept135,self.dept90,self.up_dept135,self.down_dept135
        else:

            return errors

    def back_state(self):
        if self.all_spectrum != [] and self.dept135 == [] and self.dept90 == []:
            self.match_state = 0
        elif self.all_spectrum == [] and self.dept135 != [] and self.dept90 == []:
            self.match_state = 1
        elif self.all_spectrum == [] and self.dept135 == [] and self.dept90 != []:
            self.match_state = 2
        elif self.all_spectrum == [] and self.dept135 != [] and self.dept90 != []:
            self.match_state = 3
        elif self.all_spectrum != [] and self.dept135 != [] and self.dept90 != []:
            self.match_state = 4
        elif self.all_spectrum != [] and self.dept135 == [] and self.dept90 != []:
            self.match_state = 5
        elif self.all_spectrum != [] and self.dept135 != [] and self.dept90 == []:
            self.match_state = 6
        return self.match_state

    def back_match_type_state(self):
        if self.match_state == 0 or self.match_state == 4 or self.match_state == 5 or self.match_state == 6:
            self.match_type_state = ['Quaternaries','Tertiaries', 'Secondaries', 'Primaries']
        elif self.match_state == 1 or self.match_state == 3:
            self.match_type_state = ['Tertiaries','Secondaries', 'Primaries']
        elif self.match_state == 2:
            self.match_type_state = ['Tertiaries']
        return self.match_type_state


class ProcesInputdata(ReceiveData):
    def __init__(self, all_spectrum_str, up_dept135_str, down_dept135_str,dept90_str, align_value=0.2):
        super().__init__(all_spectrum_str, up_dept135_str, down_dept135_str, dept90_str)
        self.Primaries = []
        self.Secondaries = []
        self.Tertiaries = []
        self.Quaternaries = []
        self.pstq = []
        self.pt = []
        self.psq = []
        self.align_value = align_value
        self.realigh()


    def realigh(self):
       if self.match_state == 0:
           self.pstq = self.all_spectrum
           return self.pstq
       if self.match_state == 1:
           for value135 in self.up_dept135:
               self.pt.append(value135)
           for value135 in self.down_dept135:
               self.Secondaries.append(value135)
           return self.pt, self.Secondaries
       if self.match_state == 2:
           self.Tertiaries = self.dept90
           return self.Tertiaries
       if self.match_state == 3:
           remove_values = []
           used_index = []
           for value90 in self.dept90:
               self.Tertiaries.append(value90)
               for index, value135 in enumerate(self.up_dept135) :
                   if index in used_index:
                       continue
                   if abs(value90 - value135) <= self.align_value + 0.001:
                       remove_values.append(value135)
                       used_index.append(index)
                       break
                   else:
                        continue

           self.Primaries = self.list_difference(self.up_dept135,remove_values)
           for value135 in self.down_dept135:
              self.Secondaries.append(value135)
           return self.Tertiaries, self.Secondaries, self.Primaries

       if self.match_state == 4:
           remove_values1 = []
           remove_values2 = []
           used_index1 = []
           used_index2 = []
           for value90 in self.dept90:
               self.Tertiaries.append(value90)
               for index,value135 in enumerate(self.up_dept135):
                   if index in used_index1:
                       continue
                   if abs(value90 - value135) <= self.align_value + 0.001:
                       remove_values1.append(value135)
                       used_index1.append(index)
                       break
                   else:
                        continue
           self.Primaries = self.list_difference(self.up_dept135,remove_values1)
           for value135 in self.down_dept135:
              self.Secondaries.append(value135)
           for value135 in self.dept135:
              for index,allvalue in enumerate(self.all_spectrum):
                  if index in used_index2:
                      continue
                  if abs(value135 - allvalue) <= self.align_value + 0.001:
                      remove_values2.append(allvalue)
                      used_index2.append(index)
                      #print(remove_values2)
                      break
                  else:
                       continue
           self.Quaternaries =self.list_difference(self.all_spectrum,remove_values2)  #有问题，两个数字相同时
           return self.Tertiaries, self.Secondaries, self.Primaries, self.Quaternaries
       if self.match_state == 5:
           remove_values = []
           used_index = []
           for value90 in self.dept90:
               self.Tertiaries.append(value90)
           for value90 in self.dept90:
               for index,allvalue in enumerate(self.all_spectrum):
                   if index in used_index:
                       continue
                   if abs(value90 - allvalue) <= self.align_value + 0.001:
                       remove_values.append(allvalue)
                       used_index.append(index)
                       break
                   else:
                        continue
           self.psq = self.list_difference(self.all_spectrum,remove_values)
           return self.Tertiaries, self.psq
       if self.match_state == 6:
           remove_values = []
           used_index = []
           for value135 in self.down_dept135:
               self.Secondaries.append(value135)
           for value135 in self.dept135:
               for index,allvalue in enumerate(self.all_spectrum):
                   if index in used_index:
                       continue
                   if abs(value135 - allvalue) <= self.align_value + 0.001:
                       remove_values.append(allvalue)
                       used_index.append(index)
                       break
                   else:
                        continue
           self.Quaternaries = self.list_difference(self.all_spectrum,remove_values)
           for value135 in self.up_dept135:
               self.pt.append(value135)
           return self.Secondaries, self.Quaternaries, self.pt

    def list_difference(self,list1, list2):
    # 创建一个字典来记录每个元素在 list2 中的出现次数
        count_dict = {}
        for item in list2:
            count_dict[item] = count_dict.get(item, 0) + 1

        # 结果列表
        difference = []

        # 遍历 list1，对于每个元素，检查 count_dict
        for item in list1:
            if count_dict.get(item, 0) > 0:
                # 如果在 count_dict 中可以找到 item，减少计数并跳过
                count_dict[item] -= 1
            else:
                # 否则将元素添加到差集中
                difference.append(item)

        return difference

class ProductMatchData:
    def __init__(self, processinputdata_obj,process_sdf_obj):   #用户输入数据及类型对象；库文件处理结果对象
        self.processinputdata_obj = processinputdata_obj
        self.processsdf_obj = process_sdf_obj
        self.compound_inf_dict = process_sdf_obj.compound_inf_dict
        self.match_state = processinputdata_obj.match_state
        self.Tertiaries = processinputdata_obj.Tertiaries
        self.Secondaries = processinputdata_obj.Secondaries
        self.Primaries = processinputdata_obj.Primaries
        self.Quaternaries = processinputdata_obj.Quaternaries
        self.psq = processinputdata_obj.psq
        self.pt = processinputdata_obj.pt
        self.pstq = processinputdata_obj.pstq
        self.all_match_data_dict = {}
        self.c_len_list = []
        self.max_c_len = None
        self.min_c_len = None
        self.get_match_data()


    def get_match_data(self):
        data_list = self.Tertiaries + self.Secondaries + self.Primaries + self.Quaternaries + self.psq + self.pt + self.pstq
        type_list = ['Tertiaries']*len(self.Tertiaries) + ['Secondaries']*len(self.Secondaries) + ['Primaries']*len(self.Primaries) + ['Quaternaries']*len(self.Quaternaries) + ['psq']*len(self.psq) + ['pt']*len(self.pt) + ['pstq']*len(self.pstq)
        self.all_match_data_dict = {1:[data_list, type_list]}
        print(f'your data and type: {self.all_match_data_dict}')

        return self.all_match_data_dict





























