# ============================================================
# pattern.py —— 棋型推断层（核心）
# 职责：给定盘面/点/方向 → 棋型记录（父类 + 子类）
# 对外只暴露 PatternAnalyzer 类
#
# 数据流：
#   scan_two → overall_judge → normalize → classify（父类）→ subclass（子类）
#
# 当前进度：父类部分已实现（scan_two / overall_judge / normalize / classify）
# 子类部分（_subclass / 编号表 / 特征编码）待写
#
# 版本开关：USE_SUBCLASS = True（152 子类，当前实现）/ False（去 4 低配，未实现）
# ============================================================

from utils import SIZE, EMPTY, BLACK, WHITE, DIRECTIONS, in_board, cell_state

USE_SUBCLASS = True  # TODO: 版本开关，去 4 低配版时改 False


class PatternAnalyzer:
    """棋型推断器：一个点 / 一条线 / 整盘 → 棋型记录"""

    def __init__(self, use_subclass=USE_SUBCLASS):
        self.use_subclass = use_subclass
        self._build_subclass_table()

    # ============================================================
    # 内部：扫描原语（父类判定核心，已从 gomoku.py 移植）
    # ============================================================

    def _scan_two(self, board, r, c, player, dr, dc):
        """单向扫描：从基子 (r,c) 沿 (dr,dc) 向两侧扫一条线。
        契约：每侧最多计入 1 个缝；第一缝位置被保护；第二缝(_X)闭合时
        以 type-4 断裂端截断，该子不计入 stones。
        返回 {stones, l_gap, r_gap, l_gap_locations, r_gap_locations, l_edge, r_edge}
        """
        stones = 1
        l_gap_count = 0
        r_gap_count = 0
        l_gap_location = 0
        r_gap_location = 0

        i = 1
        r_bull, r_gap = True, False
        l_bull, l_gap = True, False

        # [位置, 类型]  1=__  2=|  3=_|  4=第二个_X（断裂端）
        l_edge = [0, 0]
        r_edge = [0, 0]

        while (r_bull or l_bull):
            rr, cr = r + i * dr, c + i * dc
            rl, cl = r - i * dr, c - i * dc

            # ----- 向右扫描 -----
            if r_bull and r_gap_count < 2:
                st = cell_state(board, rr, cr, player)
                if st == 'stone':
                    stones += 1
                    if r_gap:
                        r_gap_count += 1
                        if r_gap_location == 0:
                            r_gap_location = i - 1
                        r_gap = False
                        if r_gap_count >= 2:
                            # 第二个 _X：断裂端截断，该子不计入本段
                            r_edge[0] = i - 1
                            r_edge[1] = 4
                            r_gap_count = 1
                            stones -= 1
                            r_bull = False
                elif st == 'open':
                    if r_gap:
                        # 双空！看第三个位置区分端1(__) / 端5(__x)
                        st2 = cell_state(board, r + (i + 1) * dr, c + (i + 1) * dc, player)
                        r_bull = False
                        r_edge[0] = i - 1
                        r_edge[1] = 5 if st2 == 'stone' else 1
                    else:
                        r_gap = True
                else:  # close
                    r_bull = False
                    if r_gap:
                        # _| 空后墙，半活端
                        r_edge[0] = i - 1
                        r_edge[1] = 3
                    else:
                        # | 贴墙，堵端
                        r_edge[0] = i
                        r_edge[1] = 2

            # ----- 向左扫描 -----
            if l_bull and l_gap_count < 2:
                st = cell_state(board, rl, cl, player)
                if st == 'stone':
                    stones += 1
                    if l_gap:
                        l_gap_count += 1
                        if l_gap_location == 0:
                            l_gap_location = i - 1
                        l_gap = False
                        if l_gap_count >= 2:
                            l_edge[0] = i - 1
                            l_edge[1] = 4
                            l_gap_count = 1
                            stones -= 1
                            l_bull = False
                elif st == 'open':
                    if l_gap:
                        # 双空！看第三个位置区分端1(__) / 端5(__x)
                        st2 = cell_state(board, r - (i + 1) * dr, c - (i + 1) * dc, player)
                        l_bull = False
                        l_edge[0] = i - 1
                        l_edge[1] = 5 if st2 == 'stone' else 1
                    else:
                        l_gap = True
                else:  # close
                    l_bull = False
                    if l_gap:
                        l_edge[0] = i - 1
                        l_edge[1] = 3
                    else:
                        l_edge[0] = i
                        l_edge[1] = 2

            i += 1

        return {
            'stones': stones,
            'l_gap': l_gap_count,
            'r_gap': r_gap_count,
            'l_gap_locations': l_gap_location,
            'r_gap_locations': r_gap_location,
            'l_edge': l_edge,
            'r_edge': r_edge,
        }

    def _overall_judge(self, res):
        """双缝 stones=3 的 x_X_x 跳三整体判定（normalize 之前调用）。
        填任一缝只成冲四（SLEEP4），整体恒为眠三，与两端类型无关。
        命中 → {'overall': 'SLEEP3'}；未命中 → None（走 normalize → classify）
        """
        stones = res['stones']
        l_gap, r_gap = res['l_gap'], res['r_gap']
        if not (stones == 3 and l_gap == 1 and r_gap == 1):
            return None
        return {'overall': 'SLEEP3'}

    def _normalize(self, res):
        """scan_two 输出 → 1~2 条棋型记录。
        总 gap<=1 → 1 条；左右各一缝 → 拆 2 条（左段+中心段 / 中心段+右段），
        被拆侧缝端标记为 4（断裂端）。
        """
        stones = res['stones']
        l_gap, r_gap = res['l_gap'], res['r_gap']
        l_loc, r_loc = res['l_gap_locations'], res['r_gap_locations']
        l_edge, r_edge = res['l_edge'], res['r_edge']

        if l_gap + r_gap <= 1:
            gap_location = l_loc if l_gap else (r_loc if r_gap else 0)
            return [{'stones': stones, 'gap_location': gap_location,
                     'l_edge': l_edge, 'r_edge': r_edge}]

        # 双缝：拆两份，基子端=缝（断裂端，标记为 4）
        left_seg = (l_edge[0] - 1) - l_loc
        right_seg = (r_edge[0] - 1) - r_loc
        mid = stones - left_seg - right_seg

        return [
            {'stones': left_seg + mid, 'gap_location': l_loc,
             'l_edge': l_edge, 'r_edge': [r_loc, 4]},
            {'stones': mid + right_seg, 'gap_location': r_loc,
             'l_edge': [l_loc, 4], 'r_edge': r_edge},
        ]

    def _classify(self, item):
        """标准化记录 → 父类（10 类）。
        端类型: 1=活(__/_x) 2=堵(|) 3=半活(_|) 4=断裂端(第二个_x，等价1活)
               5=延伸端(__x，双空后还有己方子，等价1活)
        """
        stones = item['stones']
        gap = 1 if item['gap_location'] != 0 else 0
        l0, lt = item['l_edge']
        r0, rt = item['r_edge']

        # 段长（墙到墙，类型3的墙在空后+1）
        wall_l = l0 + (1 if lt == 3 else 0)
        wall_r = r0 + (1 if rt == 3 else 0)
        seg_len = wall_l + wall_r - 1

        # 活端候选集：1(活) 3(半活) 4(断裂=活) 5(延伸=活)
        OPEN_END = (1, 3, 4, 5)

        # 1. DEAD
        if lt in (2, 3) and rt in (2, 3) and seg_len < 5:
            return 'DEAD'

        # 2. SINGLE
        if stones == 1:
            return 'SINGLE'

        # 3. FIVE
        if stones >= 5 and gap == 0:
            return 'FIVE'

        # 4/5. 四连级
        if stones == 4:
            if gap == 0 and lt in OPEN_END and rt in OPEN_END:
                return 'LIVE4'
            return 'SLEEP4'
        if stones > 4:          # 带缝 5 子+：缝唯一补点 → 眠四
            return 'SLEEP4'

        # 6/7. 三连级
        if stones == 3:
            if gap == 0:
                if (lt in (1, 4, 5) and rt in OPEN_END) or (rt in (1, 4, 5) and lt in OPEN_END):
                    return 'LIVE3'          # 1·1 / 1·3 / 4·1 / 4·3 / 5·x，排除 3·3
            elif lt in OPEN_END and rt in OPEN_END:
                return 'LIVE3'              # gap=1 时 3·3 也算活，4 同样算
            return 'SLEEP3'

        # 8/9. 二连级
        if stones == 2:
            if (lt in (1, 4, 5) and rt in OPEN_END) or (rt in (1, 4, 5) and lt in OPEN_END):
                return 'LIVE2'              # 1·1 / 1·3 / 4·1 / 4·3 / 5·x，排除 3·3
            return 'SLEEP2'

        # 10. NONE
        return 'NONE'

    # ============================================================
    # 对外接口：父类级（当前可用）
    # ============================================================

    def analyze_direction(self, board, r, c, player, dr, dc):
        """单方向棋型：扫描 → 整体判定 → 标准化 → 父类判定。
        返回 [{stones, gap_location, l_edge, r_edge, parent, dir}, ...]（1~2 条）
        """
        res = self._scan_two(board, r, c, player, dr, dc)
        gap_sides = self._gap_sides(res)
        overall = self._overall_judge(res)
        if overall is not None:
            # 整体判定记录（双缝跳三 → 眠三）：边标记为虚拟（无意义），父类直接用
            rec = {'stones': res['stones'], 'gap_location': 1,
                   'l_edge': res['l_edge'], 'r_edge': res['r_edge'],
                   'parent': overall['overall'], 'dir': (dr, dc),
                   'overall': True, 'gap_sides': gap_sides}
            rec['sub_id'] = self._subclass(rec['parent'], rec)
            rec['sub_name'] = self.subclass_table[self.parent_id_of[rec['parent']]][rec['sub_id']]
            return [rec]
        items = self._normalize(res)
        out = []
        for item in items:
            rec = {
                'stones': item['stones'],
                'gap_location': item['gap_location'],
                'l_edge': item['l_edge'],
                'r_edge': item['r_edge'],
                'parent': self._classify(item),
                'dir': (dr, dc),
                'gap_sides': gap_sides,
            }
            rec['sub_id'] = self._subclass(rec['parent'], rec)
            rec['sub_name'] = self.subclass_table[self.parent_id_of[rec['parent']]][rec['sub_id']]
            out.append(rec)
        # 端 5 缝合（后处理，不动 scan_two 底层）：
        # SINGLE 若任一端口为 type 5 → 它不是真单子，而是双空跳链的节点，
        # 一律剔除；从该子自身视角拆对：左端 5 → 向左构造一对，右端 5 → 向右构造一对。
        # 同一对会被其两个端点各构造一次（基子视角语义，与 threats_of 一致）。
        if len(out) == 1 and out[0]['parent'] == 'SINGLE':
            lt5 = out[0]['l_edge'][1]
            rt5 = out[0]['r_edge'][1]
            if lt5 == 5 or rt5 == 5:
                out.clear()
                if lt5 == 5:
                    out.extend(self._synth_double_gap(board, r, c, player, dr, dc, 'l'))
                if rt5 == 5:
                    out.extend(self._synth_double_gap(board, r, c, player, dr, dc, 'r'))
        return out

    def _synth_double_gap(self, board, r, c, player, dr, dc, side):
        """从 (r,c) 的 side 侧（'l'/'r'）type 5 信号构造一对双空跳 x__x。
        返回 0~1 条记录：stones=2、gap_location=2（双空跳）、double_gap=True。
        端口 = 对的外侧两端（基子外侧端 + 对面段外侧端），可为 5（链未断）。
        """
        res = self._scan_two(board, r, c, player, dr, dc)
        if side == 'r':
            d, t = res['r_edge']
            if t != 5:
                return []
            er, ec = r + (d + 2) * dr, c + (d + 2) * dc
            if not in_board(er, ec) or board[er][ec] != player:
                return []
            res2 = self._scan_two(board, er, ec, player, dr, dc)
            l_edge = res['l_edge']      # 基子外侧端（左端口）
            r_edge = res2['r_edge']     # 对面段外侧端（右端口）
        else:
            d, t = res['l_edge']
            if t != 5:
                return []
            el, ec2 = r - (d + 2) * dr, c - (d + 2) * dc
            if not in_board(el, ec2) or board[el][ec2] != player:
                return []
            res2 = self._scan_two(board, el, ec2, player, dr, dc)
            l_edge = res2['l_edge']     # 对面段外侧端（左端口）
            r_edge = res['r_edge']      # 基子外侧端（右端口）
        lt, rt = l_edge[1], r_edge[1]
        # 双空跳二定级：2·2 死、一端堵眠、两端活（含 3·3）活
        if lt == 2 and rt == 2:
            parent = 'DEAD'
        elif lt == 2 or rt == 2:
            parent = 'SLEEP2'
        else:
            parent = 'LIVE2'
        item = {'stones': 2, 'gap_location': 2,
                'l_edge': l_edge, 'r_edge': r_edge,
                'parent': parent, 'dir': (dr, dc),
                'gap_sides': None, 'double_gap': True}
        item['sub_id'] = self._subclass(parent, item)
        item['sub_name'] = self.subclass_table[self.parent_id_of[parent]][item['sub_id']]
        return [item]

    def analyze_point(self, board, r, c, player):
        """给定点（视为已落子该玩家），四方向棋型记录汇总。
        返回 4 方向的全部记录列表（每条含 parent 父类）。
        """
        records = []
        for (dr, dc) in DIRECTIONS:
            records.extend(self.analyze_direction(board, r, c, player, dr, dc))
        return records

    # ============================================================
    # 子类部分（待写）
    # ============================================================

    def _build_subclass_table(self):
        """双层子类编号表：
        {父类ID: [子类名, ...]} 子类 ID = 父类内部序号（从 0 起）
        父类 ID：0=FIVE 1=LIVE4 2=SLEEP4 3=LIVE3 4=SLEEP3
                 5=LIVE2 6=SLEEP2 7=DEAD 8=SINGLE 9=NONE
        """
        self.parent_names = ['FIVE', 'LIVE4', 'SLEEP4', 'LIVE3', 'SLEEP3',
                             'LIVE2', 'SLEEP2', 'DEAD', 'SINGLE', 'NONE']
        self.parent_id_of = {n: i for i, n in enumerate(self.parent_names)}
        self.subclass_table = {
            0: ['five'],                                      # FIVE 1
            1: [                                              # LIVE4 10（端型 1/3/4/5 无序）
                'live4_11', 'live4_13', 'live4_14', 'live4_15',
                'live4_33', 'live4_34', 'live4_35',
                'live4_44', 'live4_45',
                'live4_55'],
            2: [                                              # SLEEP4 44
                # 连四 4（gap=0，一端=2，另一端 1/3/4/5）
                'sleep4_c_12', 'sleep4_c_23', 'sleep4_c_24', 'sleep4_c_25',
                # 边跳 25（gap=1 缝在边，端型有序 5×5）
                'sleep4_e_11', 'sleep4_e_12', 'sleep4_e_13', 'sleep4_e_14', 'sleep4_e_15',
                'sleep4_e_21', 'sleep4_e_22', 'sleep4_e_23', 'sleep4_e_24', 'sleep4_e_25',
                'sleep4_e_31', 'sleep4_e_32', 'sleep4_e_33', 'sleep4_e_34', 'sleep4_e_35',
                'sleep4_e_41', 'sleep4_e_42', 'sleep4_e_43', 'sleep4_e_44', 'sleep4_e_45',
                'sleep4_e_51', 'sleep4_e_52', 'sleep4_e_53', 'sleep4_e_54', 'sleep4_e_55',
                # 中跳 15（gap=1 缝居中，端型 1/2/3/4/5 无序）
                'sleep4_m_11', 'sleep4_m_12', 'sleep4_m_13', 'sleep4_m_14', 'sleep4_m_15',
                'sleep4_m_22', 'sleep4_m_23', 'sleep4_m_24', 'sleep4_m_25',
                'sleep4_m_33', 'sleep4_m_34', 'sleep4_m_35',
                'sleep4_m_44', 'sleep4_m_45',
                'sleep4_m_55'],
            3: [                                              # LIVE3 25
                # 连三 9（gap=0，端型 1/3/4/5 无序 −3·3）
                'live3_c_11', 'live3_c_13', 'live3_c_14', 'live3_c_15',
                'live3_c_34', 'live3_c_35',
                'live3_c_44', 'live3_c_45',
                'live3_c_55',
                # 跳三 16（gap=1，端型 1/3/4/5 有序 4×4）
                'live3_j_11', 'live3_j_13', 'live3_j_14', 'live3_j_15',
                'live3_j_31', 'live3_j_33', 'live3_j_34', 'live3_j_35',
                'live3_j_41', 'live3_j_43', 'live3_j_44', 'live3_j_45',
                'live3_j_51', 'live3_j_53', 'live3_j_54', 'live3_j_55'],
            4: [                                              # SLEEP3 28
                # 连三 4（gap=0，一端=2，另一端 1/3/4/5）
                'sleep3_c_12', 'sleep3_c_23', 'sleep3_c_24', 'sleep3_c_25',
                # 3xxx3 1
                'sleep3_33',
                # 双缝 x_x_x 15（无序 1/2/3/4/5）
                'sleep3_d_11', 'sleep3_d_12', 'sleep3_d_13', 'sleep3_d_14', 'sleep3_d_15',
                'sleep3_d_22', 'sleep3_d_23', 'sleep3_d_24', 'sleep3_d_25',
                'sleep3_d_33', 'sleep3_d_34', 'sleep3_d_35',
                'sleep3_d_44', 'sleep3_d_45',
                'sleep3_d_55',
                # 单缝 xx_x 8（有序，一端=2 另一端 1/3/4/5）
                'sleep3_s_21', 'sleep3_s_23', 'sleep3_s_24', 'sleep3_s_25',
                'sleep3_s_12', 'sleep3_s_32', 'sleep3_s_42', 'sleep3_s_52'],
            5: [                                              # LIVE2 28
                # 连二 9（端型 1/3/4/5 无序 −3·3）
                'live2_c_11', 'live2_c_13', 'live2_c_14', 'live2_c_15',
                'live2_c_34', 'live2_c_35',
                'live2_c_44', 'live2_c_45',
                'live2_c_55',
                # 跳二 9（端型 1/3/4/5 无序 −3·3）
                'live2_j_11', 'live2_j_13', 'live2_j_14', 'live2_j_15',
                'live2_j_34', 'live2_j_35',
                'live2_j_44', 'live2_j_45',
                'live2_j_55',
                # 双空跳二 10（gap=2，端型 1/3/4/5 无序，含 3·3：
                #  填一空成跳三，跳三 3·3=活三 → 双空跳 3·3 为活二）
                'live2_j2_11', 'live2_j2_13', 'live2_j2_14', 'live2_j2_15',
                'live2_j2_33', 'live2_j2_34', 'live2_j2_35',
                'live2_j2_44', 'live2_j2_45',
                'live2_j2_55'],
            6: [                                              # SLEEP2 13
                # 连型 4（一端=2，另一端 1/3/4/5）
                'sleep2_c_12', 'sleep2_c_23', 'sleep2_c_24', 'sleep2_c_25',
                # 跳型 5（一端=2 的 4 个 + 3·3）
                'sleep2_j_12', 'sleep2_j_23', 'sleep2_j_24', 'sleep2_j_25', 'sleep2_j_33',
                # 双空跳型 4（gap=2，一端=2；2·2 为 DEAD 不入表）
                'sleep2_j2_12', 'sleep2_j2_23', 'sleep2_j2_24', 'sleep2_j2_25'],
            7: ['dead'],                                      # DEAD 1
            8: ['single'],                                    # SINGLE 1
            9: ['none'],                                      # NONE 1
        }
        # 统计校验：总数应为 152（138 端 5 全分类 + 14 双空跳）
        total = sum(len(v) for v in self.subclass_table.values())
        assert total == 152, '子类总数应为 152，实际 ' + str(total)

    # ---------- 端型对工具 ----------

    @staticmethod
    def _ordered_pair(a, b):
        """有序端型对 → 字符串 'ab'（a 在前）"""
        return str(a) + str(b)

    @staticmethod
    def _unordered_pair(a, b):
        """无序端型对（镜像去重）→ 字符串，小者在前"""
        return str(min(a, b)) + str(max(a, b))

    @staticmethod
    def _gap_sides(res):
        """由 scan 结果算单缝两侧子数 → (缝左子数, 缝右子数)。
        无单缝返回 None。用于区分边缝（xxx_x）/ 中缝（xx_xx）。
        基子左侧子数 L：无左缝时 = l_edge[0]-1；有左缝时 = l_gap_location-1
        """
        stones = res['stones']
        l_gap, r_gap = res['l_gap'], res['r_gap']
        l_loc, r_loc = res['l_gap_locations'], res['r_gap_locations']
        l_edge, r_edge = res['l_edge'], res['r_edge']

        if l_gap == 1 and r_gap == 0:
            # 缝在左：缝右子数 = 基子右侧子数(R) + l_loc
            R = r_edge[0] - 1
            right = R + l_loc
            left = stones - right
            return (left, right)
        if l_gap == 0 and r_gap == 1:
            # 缝在右：缝左子数 = 基子左侧子数(L) + r_loc
            L = l_edge[0] - 1
            left = L + r_loc
            right = stones - left
            return (left, right)
        return None

    @staticmethod
    def _gap_pos_class(res):
        """单缝缝位分类：'edge'（边缝 xxx_x）/ 'mid'（中缝 xx_xx）
        缝两侧子数都 ≥2 → 中缝；任一侧 ≤1 → 边缝。
        """
        sides = PatternAnalyzer._gap_sides(res)
        if sides is None:
            return None
        left, right = sides
        if min(left, right) <= 1:
            return 'edge'
        return 'mid'

    def _subclass(self, parent, item):
        """父类 + 标准化记录 → 子类 ID（父类内部序号，双层编码的第二层）。
        判定规则按定稿的 90 类表：
          - gap=0 连型（端型无序去重）
          - gap=1 跳型（按缝位 edge/mid 分有序/无序）
        """
        stones = item['stones']
        gap = 1 if item['gap_location'] != 0 else 0
        l0, lt = item['l_edge']
        r0, rt = item['r_edge']
        gap_loc = item['gap_location']

        # 段长（墙到墙，类型3的墙在空后+1）
        wall_l = l0 + (1 if lt == 3 else 0)
        wall_r = r0 + (1 if rt == 3 else 0)
        seg_len = wall_l + wall_r - 1

        tbl = self.subclass_table[self.parent_id_of[parent]]

        def find(name):
            return tbl.index(name)

        # ---------- FIVE / DEAD / SINGLE / NONE ----------
        if parent == 'FIVE':
            return find('five')
        if parent == 'DEAD':
            return find('dead')
        if parent == 'SINGLE':
            return find('single')
        if parent == 'NONE':
            return find('none')

        # ---------- LIVE4（10）：gap=0，端型 1/3/4/5 无序 ----------
        if parent == 'LIVE4':
            return find('live4_' + self._unordered_pair(lt, rt))

        # ---------- SLEEP4（44） ----------
        if parent == 'SLEEP4':
            if gap == 0:
                # 连四：一端=2，另一端 1/3/4（无序）
                return find('sleep4_c_' + self._unordered_pair(lt, rt))
            # 跳四：按缝位分边/中
            sides = item.get('gap_sides')
            if sides is not None and min(sides) > 1:
                # 中跳：端型无序
                return find('sleep4_m_' + self._unordered_pair(lt, rt))
            # 边跳：端型有序
            return find('sleep4_e_' + self._ordered_pair(lt, rt))

        # ---------- LIVE3（25） ----------
        if parent == 'LIVE3':
            if gap == 0:
                # 连三：1/3/4 无序（3·3 已被 classify 排除）
                return find('live3_c_' + self._unordered_pair(lt, rt))
            # 跳三：1/3/4 有序
            return find('live3_j_' + self._ordered_pair(lt, rt))

        # ---------- SLEEP3（28） ----------
        if parent == 'SLEEP3':
            if item.get('overall'):
                # 双缝跳三 x_X_x（整体判定）：10 种无序端型
                return find('sleep3_d_' + self._unordered_pair(lt, rt))
            if gap == 0:
                # 连三：一端=2 → c_12/c_23/c_24；3·3 → sleep3_33
                if lt == 3 and rt == 3:
                    return find('sleep3_33')
                return find('sleep3_c_' + self._unordered_pair(lt, rt))
            # gap=1：单缝跳三（xx_x），端型有序去 2·2
            return find('sleep3_s_' + self._ordered_pair(lt, rt))

        # ---------- LIVE2（28：连 9 + 跳 9 + 双空跳 10） ----------
        if parent == 'LIVE2':
            if item.get('double_gap'):
                return find('live2_j2_' + self._unordered_pair(lt, rt))
            if gap == 0:
                return find('live2_c_' + self._unordered_pair(lt, rt))
            return find('live2_j_' + self._unordered_pair(lt, rt))

        # ---------- SLEEP2（13：连 4 + 跳 5 + 双空跳 4） ----------
        if parent == 'SLEEP2':
            if item.get('double_gap'):
                return find('sleep2_j2_' + self._unordered_pair(lt, rt))
            if gap == 0:
                return find('sleep2_c_' + self._unordered_pair(lt, rt))
            if lt == 3 and rt == 3:
                return find('sleep2_j_33')   # 跳二 3·3（填缝成眠三）
            return find('sleep2_j_' + self._unordered_pair(lt, rt))

        raise ValueError('未知父类: ' + str(parent))

    def encode_features(self, records, direction_split=True, attack=True):
        """棋型记录列表 → 特征向量（子类特征编码，待实现）"""
        raise NotImplementedError('特征编码待实现')
