from SDF import File_path,Process_sdf_files,Add_user_path
from USERINPUT import ProcesInputdata,ProductMatchData

class ProductBestCompare(ProductMatchData):
    def __init__(self,processinputdata_obj,process_sdf_obj,green_threshold=5.0):
        super().__init__(processinputdata_obj,process_sdf_obj)
        self.best_match_data_dict = {}
        self.compound_inf_dict = process_sdf_obj.compound_inf_dict

        self.match_state = processinputdata_obj.match_state
        self.threshold = green_threshold
        self.get_best_match_data()

    def get_best_match_data(self):
        for compound_id, compound_data_list in self.compound_inf_dict.items():
                 combination_id, match_data_list = next(iter(self.all_match_data_dict.items()))
                 if len(match_data_list[0]) >= len(compound_data_list[0]):
                     matches = []
                     used_indices1 = []  # 用于跟踪已使用的索引
                     used_indices2 = []
                     n = len(compound_data_list[1])
                     m = len(match_data_list[0])
                     all_error_dict = {}
                     skeleton_num = len(compound_data_list[3])

                     for i in range(n):
                         for j in range(m):
                             a, b = compound_data_list[1][i], match_data_list[0][j]
                             c, d = compound_data_list[2][i], match_data_list[1][j]
                             e,f = compound_data_list[0][i], compound_data_list[3]
                             error_value = self.error_func(a, b)
                             error_state, type_state = self.condition_func1(a, b), self.condition_func2(c, d)
                             skeleton_state = self.get_skeleton_state(e,f)
                             all_error_dict[(i, j)] = [error_value, error_state, type_state,a,b,e,skeleton_state]
                     sorted_items = sorted(all_error_dict.items(), key=lambda item: item[1][0])

                     for idx, (key, value) in enumerate(sorted_items):
                         i, j = key  # 解包键 (i, j)
                         if i in used_indices1 or j in used_indices2:
                             continue
                         else:
                             if value[1] == True and value[2] == 1 or value[1] == True and value[2] == -1:
                                 match_entry = {value[5]: [value[3], value[4], value[1], value[2], value[0], value[6]]}
                                 matches.append(match_entry)
                                 used_indices1.append(i)
                                 used_indices2.append(j)
                     for idx, (key, value) in enumerate(sorted_items):
                         i, j = key  # 解包键 (i, j)
                         if i in used_indices1 or j in used_indices2:
                             continue
                         else:
                             if value[1] == False and value[2] == 1 or value[1] == False and value[2] == -1:
                                 match_entry = {value[5]: [value[3], value[4], value[1], value[2], value[0],value[6]]}
                                 matches.append(match_entry)
                                 used_indices1.append(i)
                                 used_indices2.append(j)
                     for idx, (key, value) in enumerate(sorted_items):
                         i, j = key  # 解包键 (i, j)
                         if i in used_indices1 or j in used_indices2:
                             continue
                         else:
                             match_entry = {value[5]: [value[3], value[4], value[1], value[2], value[0], value[6]]}
                             matches.append(match_entry)
                             used_indices1.append(i)
                             used_indices2.append(j)
                     self.best_match_data_dict[(compound_id,skeleton_num)] = matches
                     #('257', 1): [{20: [54.8, 55.0, True, -1, 0.2, []]}, {25: [33.3, 33.0, True, -1, 0.3, []]}, {3: [127.6, 127.0, True, -1, 0.6, []]}, {5: [123.6, 123.0, True, -1, 0.6, []]},



    def condition_func1(self,a,b):
        if abs(a -b) <= self.threshold:
            error_state = True
        else:
            error_state = False
        return error_state

    def condition_func2(self,c,d):
        if self.match_state == 4:
            if c == d:
                type_state = 1
            else:
                type_state = 0
            return type_state
        elif self.match_state == 0:
            if c == 'Primaries' and d == 'pstq' or c == 'Secondaries' and d == 'pstq' or c == 'Tertiaries' and d == 'pstq'or c == 'Quaternaries' and d == 'pstq':
                type_state = -1
            else:
                type_state = 0
            return type_state
        elif self.match_state == 1:
            if c == 'Primaries' and d == 'pt' or  c == 'Tertiaries' and d == 'pt':
                type_state = -1
            elif c == d :
                type_state = 1
            else:
                type_state = 0
            return type_state
        elif self.match_state == 2:
            if c == d:
                type_state = 1
            else:
                type_state = 0
            return type_state
        elif self.match_state == 3:
            if c == d:
                type_state = 1
            else:
                type_state = 0
            return type_state
        elif self.match_state == 5:
            if c == d:
                type_state = 1
            elif c=='Primaries' and d == 'psq' or c == 'Secondaries' and d == 'psq' or c == 'Quaternaries' and d == 'psq':
                type_state = -1
            else:
                type_state = 0
            return type_state
        elif self.match_state == 6:
            if c == d:
                type_state = 1
            elif c=='Primaries' and d == 'pt' or c == 'Tertiaries' and d == 'pt':
                type_state = -1
            else:
                type_state = 0
            return type_state



    def error_func(self,a, b):
        return round(abs(a - b),2)
    def get_skeleton_state(self,e,f):
        satisfied_sub_skeleton_index = []
        for index,sub_skeleton in enumerate(f):
             if e in sub_skeleton:
                 satisfied_sub_skeleton_index.append(index)
        return satisfied_sub_skeleton_index





class ScoreInstitution(object):
    def __init__(self,productBestData_obj,threshold_yellow = 10.0,max_skeleton_score = 0.8):
        self.productMatchData_obj = productBestData_obj
        self.threshold_yellow = threshold_yellow
        self.threshold_green = productBestData_obj.threshold
        self.best_match_data_dict=productBestData_obj.best_match_data_dict
        self.max_skeleton_score = max_skeleton_score
        self.id_score_dict = {}
        self.get_kinds_scores()

    def get_kinds_scores(self):
        for id_skeletonnum, matches in self.best_match_data_dict.items():
            compound_id, skeleton_num = id_skeletonnum
            satisfied_skeleton_dict = {}
            for i in range(0, skeleton_num):
                satisfied_skeleton_dict[i] = []

            green_atom_index = []
            yellow_atom_index = []
            orange_atom_index = []
            red_atom_index = []
            len_matches = len(matches)

            if len_matches != 0 :
                for match in matches:
                    for atom_index, value in match.items():
                        if value[2] == True and value[3] != 0 :
                            for skeleton_index in value[5]:
                                atom_index_state = {atom_index:['green',value[4]]}
                                satisfied_skeleton_dict[skeleton_index].append(atom_index_state)
                            green_atom_index.append([atom_index,value[0],value[1],value[4]])


                        elif value[2] == False and value[3] != 0 and value[4] <= self.threshold_yellow:
                            yellow_atom_index.append([atom_index,value[0],value[1],value[4]])
                            for skeleton_index in value[5]:
                                atom_index_state = {atom_index:['yellow',value[4]]}
                                satisfied_skeleton_dict[skeleton_index].append(atom_index_state)

                        elif value[2] == False and value[3] == 0 and value[4] <= self.threshold_yellow:
                            orange_atom_index.append([atom_index,value[0],value[1],value[4]])
                            for skeleton_index in value[5]:
                                atom_index_state = {atom_index:['orange',value[4]]}
                                satisfied_skeleton_dict[skeleton_index].append(atom_index_state)

                        else:
                            red_atom_index.append([atom_index,value[0],value[1],value[4]])
                            for skeleton_index in value[5]:
                                atom_index_state = {atom_index:['red',value[4]]}
                                satisfied_skeleton_dict[skeleton_index].append(atom_index_state)

                final_skeleton_index = self.get_final_skeleton_index(satisfied_skeleton_dict)
                orin_skeleton_score, orin_none_skeleton_score = self.get_compound_score(green_atom_index, yellow_atom_index, orange_atom_index,len_matches,final_skeleton_index)
                none_ske_score_change_rate = self.none_ske_score_change_rate(len_matches,final_skeleton_index,orin_none_skeleton_score)

                score = round((orin_skeleton_score + orin_none_skeleton_score / none_ske_score_change_rate)*100,2)

                self.id_score_dict[compound_id] = [score,green_atom_index, yellow_atom_index, orange_atom_index, red_atom_index]
        self.id_score_dict = dict(sorted(self.id_score_dict.items(), key=lambda item: item[1][0], reverse=True))

    def get_final_skeleton_index(self, satisfied_skeketon):
        final_skeleton_index = []
        sub_skeleton_green_score_list = []
        if len(satisfied_skeketon) == 0:
            return final_skeleton_index
        else:
            for k,v in satisfied_skeketon.items():#{0: [{7: ['green',error]}, {14: True}, {15: True}, {9: True}, {8: False}, {10: False}], 1: [{7: True}, {20: True}, {9: True}, {11: True}, {16: True}, {18: False}]}
                add_ske_score = 0
                index_list = []
                for index_T_dict in v:
                    for index,state in index_T_dict.items():
                        index_list.append((index,state[0],state[1]))
                        if state == 'green':
                            add_ske_score += self.green_index_score()
                        elif state == 'yellow':
                            add_ske_score += self.yellow_index_score(state[1])
                        elif state == 'orange':
                            add_ske_score += self.orange_index_score(state[1])
                if len(v) != 0:
                    sub_score = add_ske_score/len(v)
                    dict = {k:[sub_score,index_list]}#[0.5, [(16, 'green',error), (18, True), (14, True), (8, False), (10, False), (9, False)]]

                    sub_skeleton_green_score_list.append(dict)
            sub_skeleton_green_score_list = sorted(sub_skeleton_green_score_list, key=lambda d: list(d.values())[0][0], reverse=True)

        judge_list = []
        for dict in sub_skeleton_green_score_list: #[{0: [0.8333333333333334, [(12, 'green',error), (5, True), (8, True), (7, True), (11, False), (6, True)]]}, {1: [0.8333333333333334, [(10, True), (5, True), (7, True), (16, True), (9, True), (14, False)]]}]

            for index_list in dict.values():

                judge_list.extend(index_list[1])
                judge_list = list(dict.fromkeys(judge_list))
                total_score = 0
                for index_state in judge_list:
                    if index_state[1] == 'green':
                        total_score += self.green_index_score()
                    elif index_state[1] == 'yellow':
                        total_score += self.yellow_index_score(index_state[2])
                    elif index_state[1] == 'orange':
                        total_score += self.orange_index_score(index_state[2])
                if total_score/len(judge_list) >= self.max_skeleton_score:
                    final_skeleton_index = [i[0] for i in judge_list]
                else:
                    break

        final_skeleton_index = list(set(final_skeleton_index))

        return final_skeleton_index

    def none_ske_score_change_rate(self,c_num, final_skeleton_index,orin_none_skeleton_score):
        none_skeleton_num = c_num - len(final_skeleton_index)
        none_skeleton_total_score = none_skeleton_num/c_num
        none_ske_score_change_rate = 1 + none_skeleton_total_score - orin_none_skeleton_score
        return none_ske_score_change_rate

    def get_compound_score(self, green_index, yellow_index, orange_index,c_num, final_skeleton_index):
        green_skeleton_score = 0
        yellow_skeleton_score = 0
        orange_skeleton_score = 0
        green_none_skeleton_score = 0
        yellow_none_skeleton_score = 0
        orange_none_skeleton_score = 0
        for i in green_index:
            if i[0] in final_skeleton_index:
                green_skeleton_score += self.green_index_score()
            else:
                green_none_skeleton_score += self.green_index_score()
        for i in yellow_index:
            if i[0] in final_skeleton_index:
                yellow_skeleton_score += self.yellow_index_score(i[3])

            else:
                yellow_none_skeleton_score += self.yellow_index_score(i[3])
        for i in orange_index:
            if i[0] in final_skeleton_index:
                orange_skeleton_score += self.orange_index_score(i[3])
            else:
                orange_none_skeleton_score += self.orange_index_score(i[3])
        #print(green_skeleton_score, yellow_skeleton_score, orange_skeleton_score, green_none_skeleton_score, yellow_none_skeleton_score, orange_none_skeleton_score)
        orin_skeleton_score = (green_skeleton_score + yellow_skeleton_score + orange_skeleton_score)/c_num
        orin_none_skeleton_score = (green_none_skeleton_score + yellow_none_skeleton_score + orange_none_skeleton_score )/c_num
        return orin_skeleton_score, orin_none_skeleton_score

    def green_index_score(self):
        single_atom_score = 1.0000
        return single_atom_score

    def yellow_index_score(self,error_value):
        single_atom_score =round((self.threshold_yellow -error_value)/(self.threshold_yellow -self.threshold_green),4)
        return single_atom_score

    def orange_index_score(self,error_value):
        if self.threshold_green<=error_value <= self.threshold_yellow:
            single_index_score = round( (self.threshold_yellow -error_value) / (self.threshold_yellow - self.threshold_green)*0.5,4)
            return single_index_score
        else:
            single_index_score = round(( (self.threshold_yellow - self.threshold_green) - error_value) / (
                        self.threshold_yellow - self.threshold_green) * 0.5, 4)
            return single_index_score



#
#
#
#
# #
# #
# file_paths = File_path()
# user_path_init = Add_user_path()
# user_path=user_path_init.select_user_path()
# print(user_path)
# file_paths.select_user_path(user_path)
# # 选择模型路径、用户路径和实验数据路径
# # file_paths.select_model_path()
# # 查看当前选择的所有路径
# all_paths = file_paths.select_all_paths()
# rd = ProcesInputdata( '204,61,45,67,89,99,150.97, 140.11, 114,125.20, 120.45, 119.57, 118.45, 118.22, 111.58, 85.83, 84.79, 72.13, 53.09, 48.84, 46.50, 40.08, 37.74, 36.64, 33.92, 27.61, 26.18, 25.38, 24.73, 23.81, 22.06, 22.04, 22.044,20.06, 14.66, 12.77',' 120.45,61, 119.57, 118.45, 111.59, 85.84, 84.79, 48.85, 46.51, 26.18, 23.82, 22.06,22.044,20.07, 14.66, 45,67,89,12.77',' 60.53, 37.74, 33.93,114, 27.61, 25.38, 24.73,22.06, 22.04',' 120.45, 119.57, 118.46, 111.59, 85.84, 84.79, 48.85, 46.51, 37.75, 33.93, 27.61, 25.38, 24.73, 22.044')
# ctype=rd.match_type_state
# print(ctype)
# b=Process_sdf_files(all_paths,ctype)
# b.process()
# print(b.compound_inf_dict)
#
# errno=ProductMatchData(rd,b)
# print(rd.match_state)
# print(rd.Tertiaries)
# print(rd.Secondaries)
# print(rd.Primaries)
# print(rd.Quaternaries)
# # print(errno.all_match_data_dict)
# pb=ProductBestCompare(rd,b,3)
# # print(pb.best_match_data_dict)
# score = ScoreInstitution(pb,6)
# # print(score.id_score_dict)

