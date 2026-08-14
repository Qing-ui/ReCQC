from SDF import Add_user_path, File_path, Process_sdf_files
from USERINPUT import ProcesInputdata, ProductMatchData


class ProductBestCompare(ProductMatchData):
    def __init__(self, processinputdata_obj, process_sdf_obj, green_threshold=5.0):
        super().__init__(processinputdata_obj, process_sdf_obj)
        self.best_match_data_dict = {}
        self.compound_inf_dict = process_sdf_obj.compound_inf_dict

        self.match_state = processinputdata_obj.match_state
        self.threshold = green_threshold
        self.get_best_match_data()

    def get_best_match_data(self):
        _, match_data_list = next(iter(self.all_match_data_dict.items()))
        user_shifts, user_types = match_data_list

        for compound_id, compound_data_list in self.compound_inf_dict.items():
            atom_indices, db_shifts, db_types, substructures = compound_data_list
            if len(user_shifts) < len(atom_indices):
                continue

            matches = []
            used_user_indices = set()
            matched_db_indices = set()

            # ReCQC 2.0 stage 1: each database atom takes the nearest unused
            # type-compatible experimental peak.  The 1.0 ambiguous DEPT labels
            # (pstq, pt and psq) retain their existing compatibility rules.
            db_type_priority = {
                "Primaries": 0,
                "Secondaries": 1,
                "Tertiaries": 2,
                "Quaternaries": 3,
            }
            typed_db_order = sorted(
                range(len(db_shifts)),
                key=lambda index: (db_type_priority.get(db_types[index], 4), index),
            )
            for db_index in typed_db_order:
                db_shift = db_shifts[db_index]
                db_type = db_types[db_index]
                compatible_user_indices = [
                    user_index
                    for user_index, user_type in enumerate(user_types)
                    if user_index not in used_user_indices
                    and self.condition_func2(db_type, user_type) != 0
                ]
                user_index = self._nearest_unused_user(
                    db_shift, user_shifts, compatible_user_indices
                )
                if user_index is None:
                    continue
                matches.append(
                    self._make_match_entry(
                        db_index,
                        user_index,
                        atom_indices,
                        db_shifts,
                        db_types,
                        substructures,
                        user_shifts,
                        user_types,
                    )
                )
                matched_db_indices.add(db_index)
                used_user_indices.add(user_index)

            # ReCQC 2.0 stage 2: pair remaining database atoms with the nearest
            # still-unused peaks without a type restriction.  Thresholds are
            # applied only after pairing when atom colours/scores are assigned.
            for db_index, db_shift in enumerate(db_shifts):
                if db_index in matched_db_indices:
                    continue
                user_type_priority = {
                    "Quaternaries": 0,
                    "Tertiaries": 1,
                    "Secondaries": 2,
                    "Primaries": 3,
                    "psq": 4,
                    "pt": 5,
                    "pstq": 6,
                }
                remaining_user_indices = sorted(
                    (
                        user_index
                        for user_index in range(len(user_shifts))
                        if user_index not in used_user_indices
                    ),
                    key=lambda index: (
                        user_type_priority.get(user_types[index], 7),
                        index,
                    ),
                )
                user_index = self._nearest_unused_user(
                    db_shift, user_shifts, remaining_user_indices
                )
                if user_index is None:
                    break
                matches.append(
                    self._make_match_entry(
                        db_index,
                        user_index,
                        atom_indices,
                        db_shifts,
                        db_types,
                        substructures,
                        user_shifts,
                        user_types,
                    )
                )
                matched_db_indices.add(db_index)
                used_user_indices.add(user_index)

            skeleton_num = len(substructures)
            self.best_match_data_dict[(compound_id, skeleton_num)] = matches

    @staticmethod
    def _nearest_unused_user(db_shift, user_shifts, candidate_indices):
        if not candidate_indices:
            return None
        # Equal errors select the first peak, matching NumPy argmin in 2.0.
        return min(candidate_indices, key=lambda index: abs(db_shift - user_shifts[index]))

    def _make_match_entry(
        self,
        db_index,
        user_index,
        atom_indices,
        db_shifts,
        db_types,
        substructures,
        user_shifts,
        user_types,
    ):
        db_shift = db_shifts[db_index]
        user_shift = user_shifts[user_index]
        atom_index = atom_indices[db_index]
        return {
            atom_index: [
                db_shift,
                user_shift,
                self.condition_func1(db_shift, user_shift),
                self.condition_func2(db_types[db_index], user_types[user_index]),
                self.error_func(db_shift, user_shift),
                self.get_skeleton_state(atom_index, substructures),
            ]
        }

    def condition_func1(self, a, b):
        if abs(a - b) <= self.threshold:
            error_state = True
        else:
            error_state = False
        return error_state

    def condition_func2(self, c, d):
        if self.match_state == 4:
            if c == d:
                type_state = 1
            else:
                type_state = 0
            return type_state
        elif self.match_state == 0:
            if (
                c == "Primaries"
                and d == "pstq"
                or c == "Secondaries"
                and d == "pstq"
                or c == "Tertiaries"
                and d == "pstq"
                or c == "Quaternaries"
                and d == "pstq"
            ):
                type_state = -1
            else:
                type_state = 0
            return type_state
        elif self.match_state == 1:
            if c == "Primaries" and d == "pt" or c == "Tertiaries" and d == "pt":
                type_state = -1
            elif c == d:
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
            elif (
                c == "Primaries"
                and d == "psq"
                or c == "Secondaries"
                and d == "psq"
                or c == "Quaternaries"
                and d == "psq"
            ):
                type_state = -1
            else:
                type_state = 0
            return type_state
        elif self.match_state == 6:
            if c == d:
                type_state = 1
            elif c == "Primaries" and d == "pt" or c == "Tertiaries" and d == "pt":
                type_state = -1
            else:
                type_state = 0
            return type_state

    def error_func(self, a, b):
        return round(abs(a - b), 2)

    def get_skeleton_state(self, e, f):
        satisfied_sub_skeleton_index = []
        for index, sub_skeleton in enumerate(f):
            if e in sub_skeleton:
                satisfied_sub_skeleton_index.append(index)
        return satisfied_sub_skeleton_index


class ScoreInstitution(object):
    def __init__(self, productBestData_obj, threshold_yellow=10.0, max_skeleton_score=0.8):
        self.productMatchData_obj = productBestData_obj
        self.threshold_yellow = threshold_yellow
        self.threshold_green = productBestData_obj.threshold
        self.best_match_data_dict = productBestData_obj.best_match_data_dict
        self.max_skeleton_score = max_skeleton_score
        if self.threshold_green >= self.threshold_yellow:
            raise ValueError("The yellow threshold must be greater than the green threshold")
        if not 0 <= self.max_skeleton_score <= 1:
            raise ValueError("The skeleton similarity threshold must be between 0 and 1")
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

            if len_matches != 0:
                for match in matches:
                    for atom_index, value in match.items():
                        if value[2] == True and value[3] != 0:
                            for skeleton_index in value[5]:
                                atom_index_state = {atom_index: ["green", value[4]]}
                                satisfied_skeleton_dict[skeleton_index].append(atom_index_state)
                            green_atom_index.append([atom_index, value[0], value[1], value[4]])

                        elif (
                            value[2] == False
                            and value[3] != 0
                            and value[4] <= self.threshold_yellow
                        ):
                            yellow_atom_index.append([atom_index, value[0], value[1], value[4]])
                            for skeleton_index in value[5]:
                                atom_index_state = {atom_index: ["yellow", value[4]]}
                                satisfied_skeleton_dict[skeleton_index].append(atom_index_state)

                        elif (
                            value[2] == False
                            and value[3] == 0
                            and value[4] <= self.threshold_yellow
                        ):
                            orange_atom_index.append([atom_index, value[0], value[1], value[4]])
                            for skeleton_index in value[5]:
                                atom_index_state = {atom_index: ["orange", value[4]]}
                                satisfied_skeleton_dict[skeleton_index].append(atom_index_state)

                        else:
                            red_atom_index.append([atom_index, value[0], value[1], value[4]])
                            for skeleton_index in value[5]:
                                atom_index_state = {atom_index: ["red", value[4]]}
                                satisfied_skeleton_dict[skeleton_index].append(atom_index_state)

                final_skeleton_index = self.get_final_skeleton_index(satisfied_skeleton_dict)
                orin_skeleton_score, orin_none_skeleton_score = self.get_compound_score(
                    green_atom_index,
                    yellow_atom_index,
                    orange_atom_index,
                    len_matches,
                    final_skeleton_index,
                )
                none_ske_score_change_rate = self.none_ske_score_change_rate(
                    len_matches, final_skeleton_index, orin_none_skeleton_score
                )

                score = round(
                    (orin_skeleton_score + orin_none_skeleton_score / none_ske_score_change_rate)
                    * 100,
                    2,
                )

                self.id_score_dict[compound_id] = [
                    score,
                    green_atom_index,
                    yellow_atom_index,
                    orange_atom_index,
                    red_atom_index,
                ]
        self.id_score_dict = dict(
            sorted(self.id_score_dict.items(), key=lambda item: item[1][0], reverse=True)
        )

    def get_final_skeleton_index(self, satisfied_skeleton):
        """Select the largest high-quality union of detected substructures.

        Each atom has one colour/error state from the carbon match.  Candidate
        substructures are ranked by their mean atom score (then by size), and a
        candidate is retained only when the score of the cumulative atom union
        remains above the user-selected skeleton threshold.
        """
        ranked_substructures = []
        for skeleton_index, atom_state_dicts in satisfied_skeleton.items():
            atoms = {}
            for atom_state_dict in atom_state_dicts:
                for atom_index, state in atom_state_dict.items():
                    color, error = state
                    atoms[atom_index] = (color, error)
            if not atoms:
                continue
            mean_score = sum(
                self._atom_state_score(color, error) for color, error in atoms.values()
            ) / len(atoms)
            ranked_substructures.append((mean_score, len(atoms), skeleton_index, atoms))

        ranked_substructures.sort(key=lambda item: (item[0], item[1]), reverse=True)

        selected_atoms = {}
        for _, _, _, candidate_atoms in ranked_substructures:
            trial_atoms = dict(selected_atoms)
            trial_atoms.update(candidate_atoms)
            trial_score = sum(
                self._atom_state_score(color, error) for color, error in trial_atoms.values()
            ) / len(trial_atoms)
            if trial_score >= self.max_skeleton_score:
                selected_atoms = trial_atoms
            else:
                break

        return sorted(selected_atoms)

    def _atom_state_score(self, color, error):
        if color == "green":
            return self.green_index_score()
        if color == "yellow":
            return self.yellow_index_score(error)
        if color == "orange":
            return self.orange_index_score(error)
        return 0.0

    def none_ske_score_change_rate(self, c_num, final_skeleton_index, orin_none_skeleton_score):
        none_skeleton_num = c_num - len(final_skeleton_index)
        none_skeleton_total_score = none_skeleton_num / c_num
        none_ske_score_change_rate = 1 + none_skeleton_total_score - orin_none_skeleton_score
        return none_ske_score_change_rate

    def get_compound_score(
        self, green_index, yellow_index, orange_index, c_num, final_skeleton_index
    ):
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

        orin_skeleton_score = (
            green_skeleton_score + yellow_skeleton_score + orange_skeleton_score
        ) / c_num
        orin_none_skeleton_score = (
            green_none_skeleton_score + yellow_none_skeleton_score + orange_none_skeleton_score
        ) / c_num
        return orin_skeleton_score, orin_none_skeleton_score

    def green_index_score(self):
        single_atom_score = 1.0000
        return single_atom_score

    def yellow_index_score(self, error_value):
        single_atom_score = round(
            (self.threshold_yellow - error_value) / (self.threshold_yellow - self.threshold_green),
            4,
        )
        return single_atom_score

    def orange_index_score(self, error_value):
        if self.threshold_green <= error_value <= self.threshold_yellow:
            single_index_score = round(
                (self.threshold_yellow - error_value)
                / (self.threshold_yellow - self.threshold_green)
                * 0.5,
                4,
            )
            return single_index_score

        single_index_score = round(
            ((self.threshold_yellow - self.threshold_green) - error_value)
            / (self.threshold_yellow - self.threshold_green)
            * 0.5,
            4,
        )
        return single_index_score
