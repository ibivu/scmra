\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj1: 0.1 I_PROTmAm_PROTmBm#23 + 0.1 I_PROTmAm_PROTmCm#24
       + 0.1 I_PROTmBm_PROTmAm#25 + 0.1 I_PROTmBm_PROTmCm#26
       + 0.1 I_PROTmCm_PROTmAm#27 + 0.1 I_PROTmCm_PROTmBm#28
       + [ 0.9 _e_PROTmAm_ID.0#8 ^2 + 0.9 _e_PROTmAm_ID.1#9 ^2
       + 0.9 _e_PROTmAm_ID.2#10 ^2 + 0.9 _e_PROTmAm_ID.3#11 ^2
       + 0.9 _e_PROTmAm_ID.4#12 ^2 + 0.0663404259040145 _e_PROTmBm_ID.0#13 ^2
       + 0.0663404259040145 _e_PROTmBm_ID.1#14 ^2
       + 0.0663404259040145 _e_PROTmBm_ID.2#15 ^2
       + 0.0663404259040145 _e_PROTmBm_ID.3#16 ^2
       + 0.0663404259040145 _e_PROTmBm_ID.4#17 ^2
       + 0.0663404259040145 _e_PROTmCm_ID.0#18 ^2
       + 0.0663404259040145 _e_PROTmCm_ID.1#19 ^2
       + 0.0663404259040145 _e_PROTmCm_ID.2#20 ^2
       + 0.0663404259040145 _e_PROTmCm_ID.3#21 ^2
       + 0.0663404259040145 _e_PROTmCm_ID.4#22 ^2 ] / 2
Subject To
 c1:  - 0.166666666666667 r_PROTmAm_PROTmCm#1 - 0.192 r_i_i#6
      - 0.148936170212766 s_PROTmAm#7 + _e_PROTmAm_ID.0#8  = 0
 c2:  - 0.0099009900990099 r_PROTmAm_PROTmBm#0
      + 0.166666666666667 r_PROTmAm_PROTmCm#1 + 0.2 r_i_i#6
      + 0.0638297872340425 s_PROTmAm#7 + _e_PROTmAm_ID.1#9  = 0
 c3:  0.188118811881188 r_PROTmAm_PROTmBm#0 - 0.0425531914893617 s_PROTmAm#7
      + _e_PROTmAm_ID.2#10  = 0
 c4:  - 0.306930693069307 r_PROTmAm_PROTmBm#0
      - 0.666666666666667 r_PROTmAm_PROTmCm#1 - 0.28 r_i_i#6
      + _e_PROTmAm_ID.3#11  = 0
 c5:  0.98019801980198 r_PROTmAm_PROTmBm#0 + r_PROTmAm_PROTmCm#1 + 0.6 r_i_i#6
      + 0.340425531914894 s_PROTmAm#7 + _e_PROTmAm_ID.4#12  = 0
 c6:  - 0.192 r_PROTmBm_PROTmAm#2 - 0.166666666666667 r_PROTmBm_PROTmCm#3
      + _e_PROTmBm_ID.0#13  = 0
 c7:  0.2 r_PROTmBm_PROTmAm#2 + 0.166666666666667 r_PROTmBm_PROTmCm#3
      - 0.0099009900990099 r_i_i#6 + _e_PROTmBm_ID.1#14  = 0
 c8:  0.188118811881188 r_i_i#6 + _e_PROTmBm_ID.2#15  = 0
 c9:  - 0.28 r_PROTmBm_PROTmAm#2 - 0.666666666666667 r_PROTmBm_PROTmCm#3
      - 0.306930693069307 r_i_i#6 + _e_PROTmBm_ID.3#16  = 0
 c10: 0.6 r_PROTmBm_PROTmAm#2 + r_PROTmBm_PROTmCm#3 + 0.98019801980198 r_i_i#6
      + _e_PROTmBm_ID.4#17  = 0
 c11: - 0.192 r_PROTmCm_PROTmAm#4 - 0.166666666666667 r_i_i#6
      + _e_PROTmCm_ID.0#18  = 0
 c12: 0.2 r_PROTmCm_PROTmAm#4 - 0.0099009900990099 r_PROTmCm_PROTmBm#5
      + 0.166666666666667 r_i_i#6 + _e_PROTmCm_ID.1#19  = 0
 c13: 0.188118811881188 r_PROTmCm_PROTmBm#5 + _e_PROTmCm_ID.2#20  = 0
 c14: - 0.28 r_PROTmCm_PROTmAm#4 - 0.306930693069307 r_PROTmCm_PROTmBm#5
      - 0.666666666666667 r_i_i#6 + _e_PROTmCm_ID.3#21  = 0
 c15: 0.6 r_PROTmCm_PROTmAm#4 + 0.98019801980198 r_PROTmCm_PROTmBm#5 + r_i_i#6
      + _e_PROTmCm_ID.4#22  = 0
 Ind_PROTmAm_PROTmBm#0: I_PROTmAm_PROTmBm#23 = 0 -> r_PROTmAm_PROTmBm#0  = 0
 Ind_PROTmAm_PROTmCm#1: I_PROTmAm_PROTmCm#24 = 0 -> r_PROTmAm_PROTmCm#1  = 0
 Ind_PROTmBm_PROTmAm#2: I_PROTmBm_PROTmAm#25 = 0 -> r_PROTmBm_PROTmAm#2  = 0
 Ind_PROTmBm_PROTmCm#3: I_PROTmBm_PROTmCm#26 = 0 -> r_PROTmBm_PROTmCm#3  = 0
 Ind_PROTmCm_PROTmAm#4: I_PROTmCm_PROTmAm#27 = 0 -> r_PROTmCm_PROTmAm#4  = 0
 Ind_PROTmCm_PROTmBm#5: I_PROTmCm_PROTmBm#28 = 0 -> r_PROTmCm_PROTmBm#5  = 0
Bounds
      r_PROTmAm_PROTmBm#0 Free
      r_PROTmAm_PROTmCm#1 Free
      r_PROTmBm_PROTmAm#2 Free
      r_PROTmBm_PROTmCm#3 Free
      r_PROTmCm_PROTmAm#4 Free
      r_PROTmCm_PROTmBm#5 Free
      r_i_i#6 = -1
      _e_PROTmAm_ID.0#8 Free
      _e_PROTmAm_ID.1#9 Free
      _e_PROTmAm_ID.2#10 Free
      _e_PROTmAm_ID.3#11 Free
      _e_PROTmAm_ID.4#12 Free
      _e_PROTmBm_ID.0#13 Free
      _e_PROTmBm_ID.1#14 Free
      _e_PROTmBm_ID.2#15 Free
      _e_PROTmBm_ID.3#16 Free
      _e_PROTmBm_ID.4#17 Free
      _e_PROTmCm_ID.0#18 Free
      _e_PROTmCm_ID.1#19 Free
      _e_PROTmCm_ID.2#20 Free
      _e_PROTmCm_ID.3#21 Free
      _e_PROTmCm_ID.4#22 Free
 0 <= I_PROTmAm_PROTmBm#23 <= 1
 0 <= I_PROTmAm_PROTmCm#24 <= 1
 0 <= I_PROTmBm_PROTmAm#25 <= 1
 0 <= I_PROTmBm_PROTmCm#26 <= 1
 0 <= I_PROTmCm_PROTmAm#27 <= 1
 0 <= I_PROTmCm_PROTmBm#28 <= 1
Binaries
 I_PROTmAm_PROTmBm#23  I_PROTmAm_PROTmCm#24  I_PROTmBm_PROTmAm#25 
 I_PROTmBm_PROTmCm#26  I_PROTmCm_PROTmAm#27  I_PROTmCm_PROTmBm#28 
End
