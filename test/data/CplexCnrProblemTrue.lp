\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj1: 0.02 I_PROTmAm_PROTmBm#45 + 0.02 I_PROTmAm_PROTmCm#46
       + 0.02 I_PROTmBm_PROTmAm#47 + 0.02 I_PROTmBm_PROTmCm#48
       + 0.02 I_PROTmCm_PROTmAm#49 + 0.02 I_PROTmCm_PROTmBm#50
       + 0.1 IDev_PROTmAm_PROTmBm#65 + 0.1 IDev_PROTmAm_PROTmCm#66
       + 0.1 IDev_PROTmBm_PROTmAm#67 + 0.1 IDev_PROTmBm_PROTmCm#68
       + 0.1 IDev_PROTmCm_PROTmAm#69 + 0.1 IDev_PROTmCm_PROTmBm#70
       + 0.1 ISDev_PROTmAm#71 + [ 0.98 _e_PROTmAm_CTR.0#15 ^2
       + 0.98 _e_PROTmAm_CTR.1#16 ^2 + 0.98 _e_PROTmAm_CTR.2#17 ^2
       + 0.98 _e_PROTmAm_CTR.3#18 ^2 + 0.98 _e_PROTmAm_CTR.4#19 ^2
       + 0.98 _e_PROTmAm_MUT.0#20 ^2 + 0.98 _e_PROTmAm_MUT.1#21 ^2
       + 0.98 _e_PROTmAm_MUT.2#22 ^2 + 0.98 _e_PROTmAm_MUT.3#23 ^2
       + 0.98 _e_PROTmAm_MUT.4#24 ^2 + 0.129124365338493 _e_PROTmBm_CTR.0#25 ^2
       + 0.129124365338493 _e_PROTmBm_CTR.1#26 ^2
       + 0.129124365338493 _e_PROTmBm_CTR.2#27 ^2
       + 0.129124365338493 _e_PROTmBm_CTR.3#28 ^2
       + 0.129124365338493 _e_PROTmBm_CTR.4#29 ^2
       + 0.129124365338493 _e_PROTmBm_MUT.0#30 ^2
       + 0.129124365338493 _e_PROTmBm_MUT.1#31 ^2
       + 0.129124365338493 _e_PROTmBm_MUT.2#32 ^2
       + 0.129124365338493 _e_PROTmBm_MUT.3#33 ^2
       + 0.129124365338493 _e_PROTmBm_MUT.4#34 ^2
       + 0.129124365338493 _e_PROTmCm_CTR.0#35 ^2
       + 0.129124365338493 _e_PROTmCm_CTR.1#36 ^2
       + 0.129124365338493 _e_PROTmCm_CTR.2#37 ^2
       + 0.129124365338493 _e_PROTmCm_CTR.3#38 ^2
       + 0.129124365338493 _e_PROTmCm_CTR.4#39 ^2
       + 0.129124365338493 _e_PROTmCm_MUT.0#40 ^2
       + 0.129124365338493 _e_PROTmCm_MUT.1#41 ^2
       + 0.129124365338493 _e_PROTmCm_MUT.2#42 ^2
       + 0.129124365338493 _e_PROTmCm_MUT.3#43 ^2
       + 0.129124365338493 _e_PROTmCm_MUT.4#44 ^2 ] / 2
Subject To
 c1#0:                           - 0.166666666666667 r_Ctr_PROTmAm_PROTmCm#1
                                 - 0.192 r_i_i#12
                                 - 0.148936170212766 s_Ctr_PROTmAm#13
                                 + _e_PROTmAm_CTR.0#15  = 0
 c2#1:                           - 0.0099009900990099 r_Ctr_PROTmAm_PROTmBm#0
                                 + 0.166666666666667 r_Ctr_PROTmAm_PROTmCm#1
                                 + 0.2 r_i_i#12
                                 + 0.0638297872340425 s_Ctr_PROTmAm#13
                                 + _e_PROTmAm_CTR.1#16  = 0
 c3#2:                           0.188118811881188 r_Ctr_PROTmAm_PROTmBm#0
                                 - 0.0425531914893617 s_Ctr_PROTmAm#13
                                 + _e_PROTmAm_CTR.2#17  = 0
 c4#3:                           - 0.306930693069307 r_Ctr_PROTmAm_PROTmBm#0
                                 - 0.666666666666667 r_Ctr_PROTmAm_PROTmCm#1
                                 - 0.28 r_i_i#12 + _e_PROTmAm_CTR.3#18  = 0
 c5#4:                           0.98019801980198 r_Ctr_PROTmAm_PROTmBm#0
                                 + r_Ctr_PROTmAm_PROTmCm#1 + 0.6 r_i_i#12
                                 + 0.340425531914894 s_Ctr_PROTmAm#13
                                 + _e_PROTmAm_CTR.4#19  = 0
 c6#5:                           - 0.18018018018018 r_Mut_PROTmAm_PROTmBm#6
                                 - 0.371428571428571 r_Mut_PROTmAm_PROTmCm#7
                                 - 0.0810810810810811 r_i_i#12
                                 - 0.134615384615385 s_Mut_PROTmAm#14
                                 + _e_PROTmAm_MUT.0#20  = 0
 c7#6:                           0.0900900900900901 r_Mut_PROTmAm_PROTmBm#6
                                 + _e_PROTmAm_MUT.1#21  = 0
 c8#7:                           0.2 r_Mut_PROTmAm_PROTmCm#7
                                 + 0.513513513513513 r_i_i#12
                                 + 0.392307692307692 s_Mut_PROTmAm#14
                                 + _e_PROTmAm_MUT.2#22  = 0
 c9#8:                           - 0.324324324324324 r_Mut_PROTmAm_PROTmBm#6
                                 - 0.742857142857143 r_Mut_PROTmAm_PROTmCm#7
                                 - 0.444444444444444 r_i_i#12
                                 - 0.130769230769231 s_Mut_PROTmAm#14
                                 + _e_PROTmAm_MUT.3#23  = 0
 c10#9:                          0.693693693693694 r_Mut_PROTmAm_PROTmBm#6
                                 + 1.6 r_Mut_PROTmAm_PROTmCm#7
                                 + 0.951951951951952 r_i_i#12
                                 + 0.561538461538462 s_Mut_PROTmAm#14
                                 + _e_PROTmAm_MUT.4#24  = 0
 c11#10:                         - 0.192 r_Ctr_PROTmBm_PROTmAm#2
                                 - 0.166666666666667 r_Ctr_PROTmBm_PROTmCm#3
                                 + _e_PROTmBm_CTR.0#25  = 0
 c12#11:                         0.2 r_Ctr_PROTmBm_PROTmAm#2
                                 + 0.166666666666667 r_Ctr_PROTmBm_PROTmCm#3
                                 - 0.0099009900990099 r_i_i#12
                                 + _e_PROTmBm_CTR.1#26  = 0
 c13#12:                         0.188118811881188 r_i_i#12
                                 + _e_PROTmBm_CTR.2#27  = 0
 c14#13:                         - 0.28 r_Ctr_PROTmBm_PROTmAm#2
                                 - 0.666666666666667 r_Ctr_PROTmBm_PROTmCm#3
                                 - 0.306930693069307 r_i_i#12
                                 + _e_PROTmBm_CTR.3#28  = 0
 c15#14:                         0.6 r_Ctr_PROTmBm_PROTmAm#2
                                 + r_Ctr_PROTmBm_PROTmCm#3
                                 + 0.98019801980198 r_i_i#12
                                 + _e_PROTmBm_CTR.4#29  = 0
 c16#15:                         - 0.0810810810810811 r_Mut_PROTmBm_PROTmAm#8
                                 - 0.371428571428571 r_Mut_PROTmBm_PROTmCm#9
                                 - 0.18018018018018 r_i_i#12
                                 + _e_PROTmBm_MUT.0#30  = 0
 c17#16:                         0.0900900900900901 r_i_i#12
                                 + _e_PROTmBm_MUT.1#31  = 0
 c18#17:                         0.513513513513513 r_Mut_PROTmBm_PROTmAm#8
                                 + 0.2 r_Mut_PROTmBm_PROTmCm#9
                                 + _e_PROTmBm_MUT.2#32  = 0
 c19#18:                         - 0.444444444444444 r_Mut_PROTmBm_PROTmAm#8
                                 - 0.742857142857143 r_Mut_PROTmBm_PROTmCm#9
                                 - 0.324324324324324 r_i_i#12
                                 + _e_PROTmBm_MUT.3#33  = 0
 c20#19:                         0.951951951951952 r_Mut_PROTmBm_PROTmAm#8
                                 + 1.6 r_Mut_PROTmBm_PROTmCm#9
                                 + 0.693693693693694 r_i_i#12
                                 + _e_PROTmBm_MUT.4#34  = 0
 c21#20:                         - 0.192 r_Ctr_PROTmCm_PROTmAm#4
                                 - 0.166666666666667 r_i_i#12
                                 + _e_PROTmCm_CTR.0#35  = 0
 c22#21:                         0.2 r_Ctr_PROTmCm_PROTmAm#4
                                 - 0.0099009900990099 r_Ctr_PROTmCm_PROTmBm#5
                                 + 0.166666666666667 r_i_i#12
                                 + _e_PROTmCm_CTR.1#36  = 0
 c23#22:                         0.188118811881188 r_Ctr_PROTmCm_PROTmBm#5
                                 + _e_PROTmCm_CTR.2#37  = 0
 c24#23:                         - 0.28 r_Ctr_PROTmCm_PROTmAm#4
                                 - 0.306930693069307 r_Ctr_PROTmCm_PROTmBm#5
                                 - 0.666666666666667 r_i_i#12
                                 + _e_PROTmCm_CTR.3#38  = 0
 c25#24:                         0.6 r_Ctr_PROTmCm_PROTmAm#4
                                 + 0.98019801980198 r_Ctr_PROTmCm_PROTmBm#5
                                 + r_i_i#12 + _e_PROTmCm_CTR.4#39  = 0
 c26#25:                         - 0.0810810810810811 r_Mut_PROTmCm_PROTmAm#10
                                 - 0.18018018018018 r_Mut_PROTmCm_PROTmBm#11
                                 - 0.371428571428571 r_i_i#12
                                 + _e_PROTmCm_MUT.0#40  = 0
 c27#26:                         0.0900900900900901 r_Mut_PROTmCm_PROTmBm#11
                                 + _e_PROTmCm_MUT.1#41  = 0
 c28#27:                         0.513513513513513 r_Mut_PROTmCm_PROTmAm#10
                                 + 0.2 r_i_i#12 + _e_PROTmCm_MUT.2#42  = 0
 c29#28:                         - 0.444444444444444 r_Mut_PROTmCm_PROTmAm#10
                                 - 0.324324324324324 r_Mut_PROTmCm_PROTmBm#11
                                 - 0.742857142857143 r_i_i#12
                                 + _e_PROTmCm_MUT.3#43  = 0
 c30#29:                         0.951951951951952 r_Mut_PROTmCm_PROTmAm#10
                                 + 0.693693693693694 r_Mut_PROTmCm_PROTmBm#11
                                 + 1.6 r_i_i#12 + _e_PROTmCm_MUT.4#44  = 0
 diffeq_r_Ctr_PROTmAm_PROTmBm#30: 0.5 r_Ctr_PROTmAm_PROTmBm#0
                                 - 0.5 r_Mut_PROTmAm_PROTmBm#6
                                 - dev_r_Ctr_PROTmAm_PROTmBm#51  = 0
 diffeq_r_Ctr_PROTmAm_PROTmCm#31: 0.5 r_Ctr_PROTmAm_PROTmCm#1
                                 - 0.5 r_Mut_PROTmAm_PROTmCm#7
                                 - dev_r_Ctr_PROTmAm_PROTmCm#52  = 0
 diffeq_r_Ctr_PROTmBm_PROTmAm#32: 0.5 r_Ctr_PROTmBm_PROTmAm#2
                                 - 0.5 r_Mut_PROTmBm_PROTmAm#8
                                 - dev_r_Ctr_PROTmBm_PROTmAm#53  = 0
 diffeq_r_Ctr_PROTmBm_PROTmCm#33: 0.5 r_Ctr_PROTmBm_PROTmCm#3
                                 - 0.5 r_Mut_PROTmBm_PROTmCm#9
                                 - dev_r_Ctr_PROTmBm_PROTmCm#54  = 0
 diffeq_r_Ctr_PROTmCm_PROTmAm#34: 0.5 r_Ctr_PROTmCm_PROTmAm#4
                                 - 0.5 r_Mut_PROTmCm_PROTmAm#10
                                 - dev_r_Ctr_PROTmCm_PROTmAm#55  = 0
 diffeq_r_Ctr_PROTmCm_PROTmBm#35: 0.5 r_Ctr_PROTmCm_PROTmBm#5
                                 - 0.5 r_Mut_PROTmCm_PROTmBm#11
                                 - dev_r_Ctr_PROTmCm_PROTmBm#56  = 0
 diffeq_r_Mut_PROTmAm_PROTmBm#36: - 0.5 r_Ctr_PROTmAm_PROTmBm#0
                                 + 0.5 r_Mut_PROTmAm_PROTmBm#6
                                 - dev_r_Mut_PROTmAm_PROTmBm#57  = 0
 diffeq_r_Mut_PROTmAm_PROTmCm#37: - 0.5 r_Ctr_PROTmAm_PROTmCm#1
                                 + 0.5 r_Mut_PROTmAm_PROTmCm#7
                                 - dev_r_Mut_PROTmAm_PROTmCm#58  = 0
 diffeq_r_Mut_PROTmBm_PROTmAm#38: - 0.5 r_Ctr_PROTmBm_PROTmAm#2
                                 + 0.5 r_Mut_PROTmBm_PROTmAm#8
                                 - dev_r_Mut_PROTmBm_PROTmAm#59  = 0
 diffeq_r_Mut_PROTmBm_PROTmCm#39: - 0.5 r_Ctr_PROTmBm_PROTmCm#3
                                 + 0.5 r_Mut_PROTmBm_PROTmCm#9
                                 - dev_r_Mut_PROTmBm_PROTmCm#60  = 0
 diffeq_r_Mut_PROTmCm_PROTmAm#40: - 0.5 r_Ctr_PROTmCm_PROTmAm#4
                                 + 0.5 r_Mut_PROTmCm_PROTmAm#10
                                 - dev_r_Mut_PROTmCm_PROTmAm#61  = 0
 diffeq_r_Mut_PROTmCm_PROTmBm#41: - 0.5 r_Ctr_PROTmCm_PROTmBm#5
                                 + 0.5 r_Mut_PROTmCm_PROTmBm#11
                                 - dev_r_Mut_PROTmCm_PROTmBm#62  = 0
 diffeq_s_Ctr_PROTmAm#42:        0.5 s_Ctr_PROTmAm#13 - 0.5 s_Mut_PROTmAm#14
                                 - dev_s_Ctr_PROTmAm#63  = 0
 diffeq_s_Mut_PROTmAm#43:        - 0.5 s_Ctr_PROTmAm#13 + 0.5 s_Mut_PROTmAm#14
                                 - dev_s_Mut_PROTmAm#64  = 0
 IndDev_Ctr_PROTmAm_PROTmBm#0:  IDev_PROTmAm_PROTmBm#65 = 0 ->
                                dev_r_Ctr_PROTmAm_PROTmBm#51  = 0
 IndDev_Ctr_PROTmAm_PROTmCm#1:  IDev_PROTmAm_PROTmCm#66 = 0 ->
                                dev_r_Ctr_PROTmAm_PROTmCm#52  = 0
 IndDev_Ctr_PROTmBm_PROTmAm#2:  IDev_PROTmBm_PROTmAm#67 = 0 ->
                                dev_r_Ctr_PROTmBm_PROTmAm#53  = 0
 IndDev_Ctr_PROTmBm_PROTmCm#3:  IDev_PROTmBm_PROTmCm#68 = 0 ->
                                dev_r_Ctr_PROTmBm_PROTmCm#54  = 0
 IndDev_Ctr_PROTmCm_PROTmAm#4:  IDev_PROTmCm_PROTmAm#69 = 0 ->
                                dev_r_Ctr_PROTmCm_PROTmAm#55  = 0
 IndDev_Ctr_PROTmCm_PROTmBm#5:  IDev_PROTmCm_PROTmBm#70 = 0 ->
                                dev_r_Ctr_PROTmCm_PROTmBm#56  = 0
 IndDev_Mut_PROTmAm_PROTmBm#6:  IDev_PROTmAm_PROTmBm#65 = 0 ->
                                dev_r_Mut_PROTmAm_PROTmBm#57  = 0
 IndDev_Mut_PROTmAm_PROTmCm#7:  IDev_PROTmAm_PROTmCm#66 = 0 ->
                                dev_r_Mut_PROTmAm_PROTmCm#58  = 0
 IndDev_Mut_PROTmBm_PROTmAm#8:  IDev_PROTmBm_PROTmAm#67 = 0 ->
                                dev_r_Mut_PROTmBm_PROTmAm#59  = 0
 IndDev_Mut_PROTmBm_PROTmCm#9:  IDev_PROTmBm_PROTmCm#68 = 0 ->
                                dev_r_Mut_PROTmBm_PROTmCm#60  = 0
 IndDev_Mut_PROTmCm_PROTmAm#10: IDev_PROTmCm_PROTmAm#69 = 0 ->
                                dev_r_Mut_PROTmCm_PROTmAm#61  = 0
 IndDev_Mut_PROTmCm_PROTmBm#11: IDev_PROTmCm_PROTmBm#70 = 0 ->
                                dev_r_Mut_PROTmCm_PROTmBm#62  = 0
 IndSDev_Ctr_PROTmAm#12:        ISDev_PROTmAm#71 = 0 -> dev_s_Ctr_PROTmAm#63
                                 = 0
 IndSDev_Mut_PROTmAm#13:        ISDev_PROTmAm#71 = 0 -> dev_s_Mut_PROTmAm#64
                                 = 0
 Ind_PROTmAm_PROTmBm#14:        I_PROTmAm_PROTmBm#45 = 0 ->
                                r_Ctr_PROTmAm_PROTmBm#0  = 0
 Ind_PROTmAm_PROTmCm#15:        I_PROTmAm_PROTmCm#46 = 0 ->
                                r_Ctr_PROTmAm_PROTmCm#1  = 0
 Ind_PROTmBm_PROTmAm#16:        I_PROTmBm_PROTmAm#47 = 0 ->
                                r_Ctr_PROTmBm_PROTmAm#2  = 0
 Ind_PROTmBm_PROTmCm#17:        I_PROTmBm_PROTmCm#48 = 0 ->
                                r_Ctr_PROTmBm_PROTmCm#3  = 0
 Ind_PROTmCm_PROTmAm#18:        I_PROTmCm_PROTmAm#49 = 0 ->
                                r_Ctr_PROTmCm_PROTmAm#4  = 0
 Ind_PROTmCm_PROTmBm#19:        I_PROTmCm_PROTmBm#50 = 0 ->
                                r_Ctr_PROTmCm_PROTmBm#5  = 0
 Ind_PROTmAm_PROTmBm#20:        I_PROTmAm_PROTmBm#45 = 0 ->
                                r_Mut_PROTmAm_PROTmBm#6  = 0
 Ind_PROTmAm_PROTmCm#21:        I_PROTmAm_PROTmCm#46 = 0 ->
                                r_Mut_PROTmAm_PROTmCm#7  = 0
 Ind_PROTmBm_PROTmAm#22:        I_PROTmBm_PROTmAm#47 = 0 ->
                                r_Mut_PROTmBm_PROTmAm#8  = 0
 Ind_PROTmBm_PROTmCm#23:        I_PROTmBm_PROTmCm#48 = 0 ->
                                r_Mut_PROTmBm_PROTmCm#9  = 0
 Ind_PROTmCm_PROTmAm#24:        I_PROTmCm_PROTmAm#49 = 0 ->
                                r_Mut_PROTmCm_PROTmAm#10  = 0
 Ind_PROTmCm_PROTmBm#25:        I_PROTmCm_PROTmBm#50 = 0 ->
                                r_Mut_PROTmCm_PROTmBm#11  = 0
Bounds
      r_Ctr_PROTmAm_PROTmBm#0 Free
      r_Ctr_PROTmAm_PROTmCm#1 Free
      r_Ctr_PROTmBm_PROTmAm#2 Free
      r_Ctr_PROTmBm_PROTmCm#3 Free
      r_Ctr_PROTmCm_PROTmAm#4 Free
      r_Ctr_PROTmCm_PROTmBm#5 Free
      r_Mut_PROTmAm_PROTmBm#6 Free
      r_Mut_PROTmAm_PROTmCm#7 Free
      r_Mut_PROTmBm_PROTmAm#8 Free
      r_Mut_PROTmBm_PROTmCm#9 Free
      r_Mut_PROTmCm_PROTmAm#10 Free
      r_Mut_PROTmCm_PROTmBm#11 Free
      r_i_i#12 = -1
      _e_PROTmAm_CTR.0#15 Free
      _e_PROTmAm_CTR.1#16 Free
      _e_PROTmAm_CTR.2#17 Free
      _e_PROTmAm_CTR.3#18 Free
      _e_PROTmAm_CTR.4#19 Free
      _e_PROTmAm_MUT.0#20 Free
      _e_PROTmAm_MUT.1#21 Free
      _e_PROTmAm_MUT.2#22 Free
      _e_PROTmAm_MUT.3#23 Free
      _e_PROTmAm_MUT.4#24 Free
      _e_PROTmBm_CTR.0#25 Free
      _e_PROTmBm_CTR.1#26 Free
      _e_PROTmBm_CTR.2#27 Free
      _e_PROTmBm_CTR.3#28 Free
      _e_PROTmBm_CTR.4#29 Free
      _e_PROTmBm_MUT.0#30 Free
      _e_PROTmBm_MUT.1#31 Free
      _e_PROTmBm_MUT.2#32 Free
      _e_PROTmBm_MUT.3#33 Free
      _e_PROTmBm_MUT.4#34 Free
      _e_PROTmCm_CTR.0#35 Free
      _e_PROTmCm_CTR.1#36 Free
      _e_PROTmCm_CTR.2#37 Free
      _e_PROTmCm_CTR.3#38 Free
      _e_PROTmCm_CTR.4#39 Free
      _e_PROTmCm_MUT.0#40 Free
      _e_PROTmCm_MUT.1#41 Free
      _e_PROTmCm_MUT.2#42 Free
      _e_PROTmCm_MUT.3#43 Free
      _e_PROTmCm_MUT.4#44 Free
 0 <= I_PROTmAm_PROTmBm#45 <= 1
 0 <= I_PROTmAm_PROTmCm#46 <= 1
 0 <= I_PROTmBm_PROTmAm#47 <= 1
 0 <= I_PROTmBm_PROTmCm#48 <= 1
 0 <= I_PROTmCm_PROTmAm#49 <= 1
 0 <= I_PROTmCm_PROTmBm#50 <= 1
      dev_r_Ctr_PROTmAm_PROTmBm#51 Free
      dev_r_Ctr_PROTmAm_PROTmCm#52 Free
      dev_r_Ctr_PROTmBm_PROTmAm#53 Free
      dev_r_Ctr_PROTmBm_PROTmCm#54 Free
      dev_r_Ctr_PROTmCm_PROTmAm#55 Free
      dev_r_Ctr_PROTmCm_PROTmBm#56 Free
      dev_r_Mut_PROTmAm_PROTmBm#57 Free
      dev_r_Mut_PROTmAm_PROTmCm#58 Free
      dev_r_Mut_PROTmBm_PROTmAm#59 Free
      dev_r_Mut_PROTmBm_PROTmCm#60 Free
      dev_r_Mut_PROTmCm_PROTmAm#61 Free
      dev_r_Mut_PROTmCm_PROTmBm#62 Free
      dev_s_Ctr_PROTmAm#63 Free
      dev_s_Mut_PROTmAm#64 Free
 0 <= IDev_PROTmAm_PROTmBm#65 <= 1
 0 <= IDev_PROTmAm_PROTmCm#66 <= 1
 0 <= IDev_PROTmBm_PROTmAm#67 <= 1
 0 <= IDev_PROTmBm_PROTmCm#68 <= 1
 0 <= IDev_PROTmCm_PROTmAm#69 <= 1
 0 <= IDev_PROTmCm_PROTmBm#70 <= 1
 0 <= ISDev_PROTmAm#71 <= 1
Binaries
 I_PROTmAm_PROTmBm#45  I_PROTmAm_PROTmCm#46  I_PROTmBm_PROTmAm#47 
 I_PROTmBm_PROTmCm#48  I_PROTmCm_PROTmAm#49  I_PROTmCm_PROTmBm#50 
 IDev_PROTmAm_PROTmBm#65  IDev_PROTmAm_PROTmCm#66  IDev_PROTmBm_PROTmAm#67 
 IDev_PROTmBm_PROTmCm#68  IDev_PROTmCm_PROTmAm#69  IDev_PROTmCm_PROTmBm#70 
 ISDev_PROTmAm#71 
End
