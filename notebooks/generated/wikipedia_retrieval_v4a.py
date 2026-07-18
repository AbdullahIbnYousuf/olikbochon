# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: kaggle
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kaggle:
#     accelerator: none
#     internet: false
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Version 4-A — clean Bengali Wikipedia retrieval reproduction
#
# This notebook independently reproduces the public Version 4 lexical/Wikipedia method using only
# the official competition input and `abyaadrafid/bnwiki`. It reports the original
# same-OOF cutoff-tuning estimate separately from nested duplicate-aware grouped validation.
# No Kaggle submission is made by this notebook.

# %% [markdown]
# ## Input and safety contract
#
# Attach the official competition input and `abyaadrafid/bnwiki`. Keep internet off. The notebook
# authenticates official competition hashes, discovers a coherent AA/AB/AC/AD Wikipedia layout,
# deduplicates byte-identical archive copies, and strictly parses every selected chunk. It never
# prints test text, IDs, probabilities, retrieved passages, or individual predictions.

# %%
from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import zipfile
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %% [markdown]
# ## Materialize the synchronized embedded runtime

# %%
RUNTIME_ARCHIVE_SHA256 = "9dd66e6f70223adbaa3bd84a52f0c67d5009ea3aed8e627874cb1bd4886fc63c"
RUNTIME_FILE_MANIFEST = {
    "olikbochon/__init__.py": "a77645398a46dfaa17cdc02963e9dc72c82f264a04c15fe5de11a0d94cea5cc6",
    "olikbochon/data_loading.py": "541e889021cee70c984152ed203ce95c0e835de58b67015798a25c3241a519da",
    "olikbochon/metrics.py": "30a8268b971a5c9d737660d0529f4983ef3f08e0d225fa06155ff2d6fd9e4175",
    "olikbochon/modeling.py": "0666af66702c0a2cb7438dae3925e79c1c897798271744ec4072035c2bc552b4",
    "olikbochon/submission.py": "fff5c7e2df7b305c87e043cbea123ba578dcc400836450ad18ca03edebf998af",
    "olikbochon/v4_kaggle.py": "5e4ecc87e7dc9d1e1c2857bae212f805b682146c29540f3c235eefe629a2b6a8",
    "olikbochon/v4_wikipedia.py": "031c6884fd5436bd153f9d7f2f2d6ab3314fb180c3f45c72d000248f5e1b92be"
}
RUNTIME_ARCHIVE_B85 = (
    "P)h>@6aWAK2mk;8Apodx6~|Hl003D4000&M003`nX=`F{V`y(~FJE72ZfSI1UoLQY9m=^4fG`XIz&=x0?uo(xG%UafRtOGQvW!F="
    "=I80II7D1(RWXK{ZO<xjWYu28vdd}1#?Ivvthw<YyNikow}^-"
    "#5Z!#WcM!n@nBo`{h6hkf0|XQR000O8001EXy15*S!2|#R)C&Lr8UO$QZ)|C6VsB$;Z*DJSVRT_%Y;R#?X>MmOaCx0pZExE)5dQ9"
    "8!Sa(dI6~~$ZWh1;($+>hI4@YXei;UVlBk%QL=q&G#>n#DcSlOH<zz!T!>}dtc+c_fxrd~b-Z>+3M=F-"
    "_iWesZbBt7q6++$<w`Gn<{=f|>gv}K^SfUIuoE4Vjs@B{ICzK?PTL(XnE}@i;Mx{|TNz<}%jp1oZM6H!^gh{C!!v%IUYR9(BZYz;"
    ">uYX!4!*b0WPIPN;;k92Rn_BM(vqb7{NHbZ$0DN^Z8jUuWA6A>yr*!>gv%0>#T9GB8xHrokv8>h=w+Uj=-"
    "tyzk=C`ZsJ1~rBt|V@TjtSKei*{)y@x^efA$Z=t`lQi#bhBQ6`S|5>lipok-ELOt-RH~ce1VYVWIj9P?@y=i7Zb*2^T~8JXIZw$="
    "0%>L{dhJj&&#u7vdHHdW9R4dNj94lS-"
    "w~lY&y%P=a9=$UNLLQCkXy4t3=`Rw=zbV2;Lj+ALEM=fe$4A7bYw($hR$*gmq;lLh8nm2caqkrGmn8@)e4SrJE36bKJ6Fd0W6y!A"
    "oMd47QP$qT&%&<H8pqCMQ1;>r9XM87c@+Xol&pxnthrH{rGlPz#<bPbcMRq${?=H{q-"
    "Y<<eOXitY+=k4##2v`9h>QS4{Yqvu7UG?x)I8I2Kg3$?F!2natVAZ3-"
    "x|AKu34mVN7>a1WFt!07)jwYwm8To<WUpyumrSaawK9;1>up1tEe<2LRf!4B{-}1-s1VM(pJ%lpxw9S568dmej|6Tw+6RhPL-"
    "+V3)2l}M8rG|{k0@4I*$C|0K6uDqkUss!NdA(1yHbOc~E`Mw}-X&I)#5FJyqDXR8HMO)cdAB4-"
    "4X{swK`OmUyg{Byx_Z<+cN_!tZpl+6ARkAL<L5C{rh)y0>0u<#w09bJ5eEwn{kMBdRCLp3rEB!M0WM{nNF}`!I`sT|+rWP$!b%p!"
    "36r}hemjy40HA3IQVVM#^?>0WRKa3-"
    "4CR#Hd>;v7_J{&JSjS|}qX+EmF^MvzsxjzXf)8mN_qM=sM1ExnaKNvCe}2!+a19Iw!#FIL<mq%wCeMTB@dh{-"
    "cH!r~KQS4)@F7^}8S4)ksE$+Q)YE#huLo2ss+kU<2hiA)GX(c0=no3i!LKD=GC}8BoIV^4VSwJe;K5$xhg`sfwI+ws|33(G>k)i7"
    "ou`oH|0GeTBt^%g1f?J<IM>Tx4b*UG)w2xDUk)bBx}D}xV8&uL5VPLA)iAsQ^!kMd<-rLs26+h$p9Xop!1I9oO|GHO02=@PUD-"
    "C^nV=y!sX)zKwIjCC@C@60A#gw6-"
    "CX;lrB^bH13)kkA5&jTe=taZIuy0;{C`W44ulL5y{o5VNAJvD#(G2;+;a*`>1TuuXXgustYP$8xixwT(1VQ}Wfxl9`^2>FJ*muc^"
    "8iM8QbInC59@u?gU6vUr02MR13`bI?t8JIH^&w8_W+LrWUiUZfi)c*c{&1yT+&yZcJY#@LFCUv+qOww+zQ}o#$~v3ENSD)lk?ZF3"
    "7YMY=8!xjc!JMSuo=|yKw;YzAW@*<-yw({k$tP<rH2W;ojIjXD!5)(-XD-3x-"
    "nO#u=Lg5Ji1RDc2^e}X#GD>O9KQH000080000X0Kh|eMg0o^06ZxG02KfL0B>w*YhrI>Xm4&WZDn+FX=8IPaCyC2dvn{k5&vJG0_"
    "DujkSVQ`(x%h8)lEH#oy#<FZfxJ9qtS365|nu6Lj_33itB#%?gAhPkd$KIU7N`y61iCH^S6tIJkR^UYM#X+B=wdvv90oi2$r+B&Z"
    ";sbv`omhDw(JW{=*WB?2hJHLXmFjdER7_@~R+Fls0w4Srn11*j2m+7G+gSGNEbWD%Tr`DP6}c;WhhL!^)T`BH&EpoC?82OJE);ih"
    "94x%1!(7tlUo~`g7S7yFH<Tl)LuP4m5!W@OPJ}<>EGHl$X;&8Ni4e#|@|PJ`!=oSqQ?GX(K?^sGvZ2ACh#g-u2j23Ckm6I-"
    "^w0LH9Ic$v-Z@Xxai*p53mi7%Z#SS$bTZUB3DFA-cXfyEzZZy2<h+k~AU7=^D1!@oG)iS)SFP@Z{#(tMlt`KfZr+O%|Z;lyX{bnE"
    "!GXl0VNwl7Nxe0-k4O{ii<$<ZCjY&8D--#k-%*-$b)lGg#%N**#0}i#Pdj_WJ5$bTN-"
    "Ue!Rd}1&w(XfyPypdXv}h&#td$`gL^v(|6}r?>?Mg-kiNhTIrgZCXU$sj`6HuWleK$^5*>F?ECjOkr5XZyaEm1{u5T(n@$E4@6J&"
    "*e?s7o5j&J$?S<;eVF|-"
    "Y0OtC;i6I{x@tKv28{V*BA|DZ0bHzlH*Qb&hMD^Kb!`X(`j8sj%YidC_t7%q3B4i~gSW)pkNh?kSO&L+41SIH5@TtlRWYsw>V<M7"
    "{oS&k<ASyYhdricWOy$MkU0Purs8Bs6b+gOa5{Z|h<{<|EDkRhCboE6YFJ&6P{-"
    "LMn_hnr*@fP#~MHxTA{cM4VjN1VQn0U8UJ*<TyhhQ@tj529n(OT>9OobDc+Pg+5s<yH382;2?jM{Exq24Kxj{<V?jU{y#2)kXr2W"
    "kd@Q0!Qp$t0JX5Z>gB)D`*8m){rjKw4wycga%csgPe>`K)}E!a&AKE)a7KFNr_K?VE)X+&7v80dPk?zR7!SBpCq!MzMtB*1G;asv"
    "$ru<G>;?%|1e-_8v?7xPCA;dQJ1V$uU`I4pxC>889H!<WLK>pR#&O03h-"
    "#&T4JM1ZcZRLXIF5kU7>;k`7?ajyu?}w*+5|FVS{=kYA({0A5m@nWy8yZ_(bD;)LBc@UhaVya2TQ%m65p>@LGvv6uxxIOIBi9@qJ"
    "Ct}Qk{g(<yboNm}cE%J^tZcCtH?AF8qduMUKUILT;qY?R}!eX%mL(W$we!*42w1#3SYG3d1s6*LAdsNeP&JM-"
    "!IPUgPK*sbgnZ5zsyx_E8CZf-w$7wKfG4b;%zIDKF!?Q%jGC-!T3s}f`_30#McM#LMW|)YEH-JH?js;~(h80BZvPC{_Z31PctMRA"
    "C&TT3!4kfjr#6NA`JP9TB!*GZzhG+K)RhrV>j+KeK<%9E0+S70{fz|O2hvarHYOR*m!_|=`^RXrK(Ir+EnHA>Q*a9m%%mVYQx8Sj"
    "7F%39&todlc7NAvq?5d+LgZtE35E6ksCEM&z@e@SA;pR$4wluW~>1(+Kp;m}VIK~K17?l13qFB_>##B6k4h{ku1Co*w_D;;+T}-"
    "zyzaS5mheO+cp=D-ek}M@~Dlm(unGnrduv#fRg%d(klp-LQ3i1gncJ;mse@-"
    "*O$PbX6?40w8`>A(h^pPt_(ZHc)&0wEWE|zB*fFev&@*to?V~O7i5IlN8D_E-"
    "{SuDDS)7o13a5jB4w!m&lyER%L%<;|=eG?^Mb_%G3pF2Y;On@VTyU*c9cYN1zXMAy2!H(G63H20O$lJ2|RNCEPEAT`6>396`uUPq"
    "I7f8S}q)N%dSkM3H^>c@ow=&FBKp^OCO1~9<p(1B#>p)J>OHfVL!iS~^T^=EzGYcKHjk>g@A_gpjuwN`MDCi#&{ojp5Q2f_Q^4qL"
    "Tpr|swVaVQ7Z=7UTNT;NqW>7?yvy~&JN&Xtos63I!Ckv^Sur<4{?FJSsDtolive)lw!D<+<5)OukoVToG=pG{hd8PV)JOnzWfj}7"
    "tK(MNzJS`9btz6wTZYb(lvnyY;tys1G8KNNgC9Ph32eiGDwO>Y33bA+v2)S+2G-"
    "nItGq8OEwGOna*}^_ww~eE%p%g*_hM%)iwJO1(##|m!MyonD#=54PqAsZ+VAy>~fHB}u`%R8g)RzD&J}a9-"
    "y2Ov~rcy26GHOU#sX!%O!+FE%3vg5b`&xqoKXBhpQ<vS+5@n@%JI}iHQiHy|24e|d&nxUbEDcw>-2=G|V~3|h3>yr0?F-"
    "Z$Qx^w&oFNx0PkJgor>jYIL(@6;*LFTHUL5G@lxtj5_AL+`?m+s-^CQMOZYPaw21)wl7S~W6-"
    "F;)$poKd%OXzx?CqkZlGY)%d#PH<iC{%cOU?5^ze=Jg_kjYjlD*-PhIM$8q__4ES+yRR8OV|0^)f*chY(bpq)sce!c%%e-"
    "m;H$hQ*z^ULN_-"
    "8<U?ss+UbiCMfW`abG()oiGEwuRy^HkK91a$K=$}a0=GJZ*=Mll0K1Y!9;sV+SC1r$hWfqC+aA0d&$cEyjP|Ot=wd-eVbpp;L4&p"
    "Hs!xa<k<bF=Bjlsba*)M%?hh{FI9_z#9P}vp{m_IXXH6y9(Q<7WWwdBpFwczzrH|AX=_7PeYX$8O%Z8zCdCMQ2gvV=ek4{!&i~Z7"
    ")YUS+Zycy%?7Ew!~)o7I-"
    ")E!HzkPJmqtW5ECmRpuJD6<$IXPq_u5ssy5IHpn1oqQMVhmk1ha=WeItwcdD+m8Fmq8W5dcH)F1xz}+~#rn%FleJqolA}G_5BWc("
    "sE^1Jl0(L^m^d;3|G}QWx_RnnKLEn~3vE;@_p{q4*PsJkzSKYI-pB14{DIBdzNd@E;BYGh;@xF5Z`W7+_-"
    "MX2gYVTQ)4^ORNr&TiInT-"
    "S6SQ|IKS7V9?s^OQbHyPZi4jHo(s_KAsufd_eP%eanBg=Bn;)cub7VUPHbdVW((hQWb0Y;%qhd4Oq1}LIot2dDAu^P$baxY1(3<H"
    "&0Tl9UyLFM3P&GF7cJWdXxB9D;fkmK@FAE5*VIQ=Z-"
    "!fci*3*LC`#}K5dL1I@Qf=!9Y^`pEPLb8Y+Y>9Y+${0TkX(rbkuU#uqx5*lhp2kmfO$KIXeb%D&I<oWC*ZA15-m~$4vVbetkiX8-"
    "v!u&^B&>cy~@yLmumDVU7z&s0lJin@Mi^dCs_1}a%{_eeN-"
    "5^#R|!bP!2TZDFJMe6<KZ1O8zoahXXuA{F_`tgqgJZ5xA32TUN@GL3it<u2%)hUFUnk?qkLhp|1?fG}|;dB^6uBnRPw*>^rM0?N%"
    "L5E$eFK$?5f-5#~2ZF-wF&nn4=~jLN#7W|VI*nH0aEM=MQ7+Urh5vvfJF$k#pV7^C&r{B5C~-"
    "cLXDwH^0e2}`{n&#o@tUA{fFqK>ptCsFkQW9m_-D0y&&d&D0GKs8`!E6-"
    "y4;L3J7oqzR+b5Xmh`$fg^8VZhb4@&Uq?5oFMO3uhO+iX?zoroQa5d<vu-"
    "o+eHZI6D8X(Pb9>%ERuBRcd_TgDZ4AOWZVkP<P^P%L#PtLK_3>Ga(^g(E<n`0h=;$+KQrI#{&y{a1HE(G&&c7}z4FIaMcV=jvEqw"
    ";jA>Ll<By`tHoTy9MeEz|GywfjeOZ6FpQ$<;^Sy0vEsgOFinXJ}q$(sEjcIU-"
    "3X;TWJ;ax(pBOXtP7JKAjs_GpSPd61VGHa1}DZWzsedeu`$j;Z?J9?+u^xi-3@9sfh4m1-e%2-"
    "MXss)#S;&iv01D{1wfxVOitQgcs1NeP-"
    "$o2etxFZL)&>AAc^Ok~e1V3QPNT8LHk!lN>4?wg6t6Jx**t0bEjXi{A2mPcx1*Zg`;)0@}&$HC^ks|1qyEsMl~VrEba${G(f8+{i"
    "ksKy01%<U{e?NR_zj$!^BS^;6Ll{ve*$RlSY$&-"
    "bupfup&SBEWL>vPbRw7$MqKc_yz{XJg?EP%gKn<=*d|zCr$JvrFXI{MZRVs%`!JK-"
    "<AGg!Gz9JFo!owV^@1o%|P2O9KQH000080000X0H*h+&UOO;05}N%02TlM0B>w*YhrI>Xm4&WZEs{{Y-"
    "w(1E^v9}R!wi?HW0n*SFk!sSVeH0_F_OS3@2^_B%6ifqy-"
    "#<prw&bgd$avTE|`NfA3I|B|EW!qCJ)eo8o+MX5Q$HeBZxQ=__Q!luR2SlP^!CP#Fl4Dw*?jrJ3bQl3Z!>4BEK;W$gRD=jB=zgwn"
    "jSl?F-"
    ")FG{5?VNxnv8<|!oRYE{opTumH?iC;5&kAG;p4T48swj7a86wNZ(7Xy@TE;oB`VM%rn$?bCi~GH!dCs%^8E;UU{{{NkR&a^`w8^P"
    "8xKnFxEKg^!)?i#t$2KJ|!P)Kgr_KFO5;Y_8rImV~HyeMxez>QP^YQ$KB;?1T_xr=_n%<B9pwpZ2{AqUch{wYL9SjEE<hSwcz38e"
    "D-97wz`#8Uy&<UPh#shSnK~7c`7g-"
    "Z}5R!|ZyQ@bY!G$PHG=>JmZh*iWmWEW)RubXdDxG~wH_X)wG?`B?Zm*|Aw26~_hBcDTS6gC(_a9~()Izrw-OjDx=&26I-"
    "RMUj&fkIa(a<d>nb@H;Ip@VKxz?<pn#naJ!DU2-;c2~sz0rL-$p<{n*l<a+Jh?pY(MjXMX}PIZZJXIVnd=Hps|mFQoiYQ-lnHZ2$"
    "ym)x%F-01HOXfs&*)`#S3)AoI2Vf9tKn%at6o9Ul38p5nP@-wr3znGFWXNqt8-"
    "OWx=4I6JRyEp?{|+UTB7T}|K1nQ72Tc<`mi;wBJ$HeX^ju><NvEQ{uR*6dKcEesbI*RIY)X)PEcx(=p^a+t#@UV`09`qysgkdAg-"
    "z$`z9G8JNC!*jdTlqy|IbEVuIoLIHU{gfJ_zC;QWE)kes7~xHTaDvg6WV5%~L_@S~0h!-"
    "xcapCU^7hj1CT3@z9S1TD2%u@x8G?t+@FN1gYGxJhM3-"
    "LzXEip!`m91$*UP3#Dk%$U|}*HQbd3A6Xjk)7QbW)iu+6g=g&aYMG;Zd8S%*O&_1b=`z|fpfPiBtb?6-"
    "4Xo}YIU%&WCdhcG$_sDFy&=$P!9$Sl9@Wlpx269Ic<ttFq>}n+PPp>Dv2+!3OYAQ6Lh1_v4kLl)+!`F;P@LIz{&BG<e7;IZnRc9$"
    "o>1~+oW~&t=+^{32eV0{7y23oo(IMi)CE41^QP1eQ#!@*b#+5_($E)RHD?7@zgee1~qcwViY-;_AxF7%W&!a4Nyx11QY-"
    "O00;m803iUj51{@h1pol!5dZ)f0001QY-wv^Z)0e0ZZC6nVr^-"
    "2b7^mGE^v9hS5a%@I1qmKuh99)246_w?j8a?=(Y(2ns#Zr%R(tewv|SlE&C)nO|IAfej~}U<s{xs+vQAYENQIyH1mBL$@l%G<TRB"
    "?)JZ{wpd~|T$%N!}Dpdos3~g{uGpw2;-}gOl#mfT4@v4?JCvl8uQI%YRDJ!Km6OBz;<~h;xh!IXwYq-EwMcGq?mgMi6u#|XSGs5b"
    "k+950ut87;V2%rQ0s;rr|%;{fAnXUm(K@@ypXAV3B&*k;SaygD?XYuXrES`S*JiVJQrZ@K&*O4Bd&A&~r;_>fe(_CC!-"
    "rY9Vkr#TNcRyX;$CtO)Ulun@bdCa_W`2Zx2#h3Ksa`9<QOP)As=<k<V2MOELvQ)z(_+3{&TnsyOp)V+<b9*r{CawGv6w2emjgf(<"
    "aenVrDEgnt)XWSujrO!s@pLg4oCmnG{-`WW3sIXrv+gW=glmy5EDI$maX~MmK0OYOCCVGCi*p;XtzL>J4^-"
    "1(DRxwB#Dp+(uLt8j7pBM%ef6La||X(t_k8m3(Da`6>yGIX=vwVWM#QXVl`j9;<zAz{yzbJjILnrnQDdT*FVus$w-"
    "T(!AlGfLvWxkdZuz+))Hk@R5{)OFBLdJ3d|%;ah~rGWuj6vh9Q-IjYiE%S4h@X-T)W@+v|df5dCtFTm~5k>ftIL--%&ssGgNS-"
    "72>b2MxKI9=+sL$iQ_FzD8(O)}T}~)`#=zx6_6ewzhasoO~Qn!Eg|cFxv&;=}`T9g=u~zP*KARC?6#RX=C=}hfTI(3W|9mJWeRXd"
    ">3o_i@^Noj4gp{MCf!SWtmSbLNy@_<BE>tt|9_pwv%P5r<r7=8!m7HX=P}Jb3r_(L(Oi#p#U;?7)R*r5f+~3K^X4StK+80O4sfa%"
    "}7j>k!J3a%z(fJ5F{S@8vUzD3d&bMu=xzrmTW-?mIM(*9yn@t&j|7I8AfU*BfVcYaY82Os9E#@Yu$atq3+@S^=<+$-"
    "J)XHtJ#t?%K>c}#vb}SO52<;2en44x@`2{LSF}38$AQAz%pGM$+aVX3!!uWG5I+DhedS6rzs#m!>ysk>7Mm!WvTO}LvO?0c7AaA7"
    "1U9}H_Z+W1TT3;cn3y-"
    "9N18fMCd0(1tMalv{BVfps}hAXKH)10ovC$qCx}hMBLZhzWvG4J8dC)dbhCZg{<+$ZXh&^^b)NgoA9~<+lTl)oxKy)u;kR~ppWSQ"
    "&)cZ@PK1_zC@YD{r~?ux+m^Vbrg?TyE(Qw%kUXQRsK>Sl+C%UIaP9H^Rl_xCtH&n=4o+IOtYz*JpVt)D+^EhRZkm$bY|d@TFi>O6"
    "Io^R+R*kUGjbMK{J+XovxT+JkUYkiw)sD1<QP^AL`q?_|sa2C4vsSX&h2N_kE`<NTdN`p!@1?6Nn=gvmla9_RH}J+XvJEm`R_FJ;"
    "CSiMznDjCONzs8eLH*;m@YU<8=dpuL_Zt1!<>9vAp02v?0a?Tw%qbkH#cBFbSH?Km#c)ms&`-OGy)-"
    "^RPt@1U2SlS>nHgQv$9?_iGgU?^0{@sgwQSl`*@nOgi@`@3`bxM??pg0*W0vinz97EAz%n%{)%dad;Ts4#d=6_oLA(!-"
    "VgjzvBs=c=;5cUUIT;F~y3QH~vXU17@E_PEAm!-%M`Y6<t<){72@6CY9YfUxq^GUpTe5xnXrkWOKw2K`%VXdCfO64ZkWNwF9-"
    "bkgxiRnXz-k@oIfAsN-hWU_0|XQR000O8001EXX+$mG$_fAg=Oq9D7XSbNZ)|C6VsB$;Z*DJkG+%3BXJ>3>E^v9R8vSqExb^q^6`"
    "b~qj5%s==p7Cje1O-uYjAB+ByP4LD+pPl9A0Efqo^cqk^g<~k(5M<l6^g_IM^}yzWBcQ-"
    "s1&9aKrNKG%19NxnL>zHQsJ>MhcnSvr5TW@j{TSkVM@xlH@EF<d(?_SKpj2iDacL(mFw~C<ua+lS~#nLTOg3S~5xr-"
    "<5?_Bo?Ajnp$;oV%L7F3eo&s=CJ}cJA+@w>OSY22Kf{GG#iTVShHF(nbD-Eg<{g~?n^GV&HhsCTUMgpm3tCbM3gPZSfsE3|CMR8R"
    "z1K8rLd=nc1%g0R4qfvQm&5*Z1X%WD@LCH9C%uUs^08)rL|*mHZ{*v+O5yYW1RCeR?Jzn*pF}M6Mx_(aETlI@Pxp>cdO<4^Yx0pz"
    "54L^<K@j<FJ4_={Nv(s`GKCVK7F|Q_v)Npuh!QWt6NxFzrSAHyubQzZa}Y>m*-"
    "a>>CJk%UfC5N)RMiqy!iBKwWe?1FRzzx*Q;y0{?ChFFX;Jly}W_Vi}R7i4;OD&mp3c=akXCJ+6c(?>fOldEik#bx^#9vF8@XCp&R"
    "y%X0FjF?2$>cp7JUw;1>n$Wjtq<S!ST-"
    "8nyiRCN?V>R{$X5obHO0<z_K2;uMUzE1BZjunN51*($P(Cme9{c*Am*I$Kq|D|6S1&R%Oqpe@VV%4GeALdb054bOpNBW@LEj5bHv"
    "1JT_smV2|;I!sYy4(^`DQt>QK)Bp^RFab9^IXOvLMyTe8@kdclEYzQ87&0?*`U`%aX`(zMyyBu#u}Cb`oZum5W*`4a2B9v<H%J%Z"
    "1KZEeeDYke-Atzu0GJc}Nw^?-Gs0h0ID46980H+TLR1Y3pZT2c4)E)-`ZCQzlo5wSvSrez4;uz-"
    "b`U=U(<{gYJ^{URltc&$oFlWjL70<GQRK53Bx55!;E>BgFps|8)r&Ib6(j3?$yQPpGR%U{0+rKC+Sa>>HTpM5n^uTfM5%~(khfog"
    ")>ob+$83cvwwNL2=QEv_=MA+p?G|_X4B$Rrv>$cSf-N>YO_?AI@-5(LFemU?;g@31s`=Inz4$N-"
    "P2$x*W${!6FDEAaLBhCBxf;4~Us1oukFA&oT(l2s={9a3<;4?|;S90b)A|^lm2Q#rz)Wzo(;Sjp-Wmx=gNfktPZDi1BxTO7{k`2|"
    "`FBN8bMD@eeZgH$C*6XT({M+>h>X9WHN%z&#%7elC0vDGur?Itgoz4^hPX;NU%ZR+ieWr}8r9;j;An6kQcy&-"
    "SW9qK@)r{PM+Eck*di%Xh`&WptL*e=yVJO_tOb40#SLBwMlvqST2WaP$~ZIlCVqv%D+BpT0|YZe>8X$pK>iTC1wjn$g#C~(I)y6f"
    "x`rwWBk_r=`=VBGX>7MxxnSxhs5v+{9-"
    "C(z;&zT1pz{nT8g$8pM8VQhLHV}mRYl><6Fk~Iq+CJ(OCWDF33`79&!EMFQS6DmF5xZ(dfxpSc3PYDZs0vYkKbdVMzk&>F`Wj>8V"
    "X7{U=pL)%{5D$tJq2aRa$$srR48SY~e>&4UjzEg9f9(rQW(kaDzdm1#a_{<1ycQP9C^*03I(m>&|xP?|zy0(Ys6EM;`1AQQI;D&u"
    "pyM#~%Zj=AYcx;$e);ILI_11vPt4EB=*D(QzQA$)u2Sw52H~h!MfOWOEF;hu|=NX@efWTDMw^S98WMqjnkk9WIO`nYyh(9M*HLL("
    "8Tf1z^8O4GqGqCl9qaTCx^DYW@lvxI4TGccr2m5qz1sRUXF2qF3a$Nb?ak<H2#;(L(3WnPLZ`W2ii;ui-"
    "P2Gtu&%4gFg3ttq(~B)XKUd&#Q%B2NLYJ4rME8*zsUI3^02Hb-"
    ")3O*zA;2)j6u1<igqEJ4T6_JF~c5vf{Q2PL?NN(9pDRu*;1rUYnrE&q;f$7M*kiwuojjZmcelD2>A(3y&rDv5LGsfcclz_$kNQs*"
    "|UpLr{tak{+`do8}BakawznXK_<#4!b28dwKoSmMPYa%k%_xa$bhSCHxLNXBBEKg`M7+qu3%TJBIWCfs!B6@k9{D-"
    ")6Cq3_<sct>ax1^zj+{o0`UYcRbHi|Tian&BOm!Zx&Z-*zN37#uGv3|c;}Z|>1{v$JE@>H5sOb?X-"
    "GZ>M+u|CtvX+onlH!PDqGR`ELtok%#9jv)I`tPP*R>!JCIWh}NV>?b3VVr#Q7z1>f5hS@3e5>XR{xI;I4OmPDoo%{f)OVMWKOQZ^"
    "_r%d5trfy@dQ6~~Xzu9jG1v+r8?(p;Ia3FM*JMU-fS*!ihRI_fSNM4Zu?iF8<D5)L;&u(~r*a#bHrgDEaWxTcX<YZ#n=olYoPN+a"
    "AR7@Gmhp8+|Q?t$|6zd1ZsPZNS3#W%E$Aub=&HW)=`{H)Wo|L4~{3xR_GuXrcPNwyQxnhWn8RVA*c^&;Yi{eTT2wI0W3DC`a!y#w"
    "bsb=nRoBPRgU=4@Rc)pw@(fyv%atG0A)>*Tv9MBn4qV8ScF#1wO6@q?h!T$>lt$$)Or!g0>ZC_nWARj1(A(d`wTiSi=hE8XC0Htx"
    "wOw0E3dIcNyue=QV;@Dvt0&5WvDttXBKR9jw1n&qu8T#e8&Q+*!=j3%$68C4`-"
    "4!!75f6uq9ChG`X%XlSSl_zte8&mTW(2Zua)0n>23Zi>0pL63Y5+a3s<8+P61efTE0HEfDw0Ts9pvALD3r0df}cJ7qLvA3>O|m!-"
    "FHgMdIMD!g;qsk7xX&%>E|DhV>P3OR@yfXj#o-"
    "I9T6J3pP?Nr9il@>U6Qy^ZL0!6gDVp>5@Vi{EQ8JQxqJuAk~{{x&<0B<;?YNdX`Jr()ARlo+o}wd%|xX&8Y7({^7t!5CxdM&@fM$"
    "Grhi`-><|LMB->(Y+U!B$f1nqJTD2+G)2Sx-SkF78S;n$L--"
    "$&b05~7C{igAZtbaOtzR{qjgW%+X4Uzr50kUMjnKo$*dt`I6gQ;ciG50qI=yeruz{D8jiDT*}HY^>xUAbZM;zSH|6&c^w=B?9@8Y"
    "nE&Vn;E%VEp?}X^t|yVwkq62)$<d(ao_&nHT$sM)&M{-"
    "uzmUmEa|GGvESHB!fL9iZb=e*J7bFLa+KvZ7WqZap*tIwKx_O?}B<K5M%V=veSGX!zGI|&fX!R>KL~a6OUXLV#l<Z2X-"
    "p=kb%*D%~Nt@R+=Of4l8k8t()$FHb9hmL58oyNHskGRVAP;MfKw`z}jczaRm23!{G!ZbG|cvHa(nh@CT!XarFLZ!NGPW1im_*k-"
    ">M)6^wGBRhRq`FS%pG2@lk3x#YGz<usWd*nk9k2FG`@DkP+b0}_pb`+Xet;CdwStd3C}gC2JYTZoU=X6T0FI|m313p1Ty`%4aXTQ"
    "K<^(G|t$;{6Ia!jf({c^h=3cpYQsY#YQe6t9J`IHRr27y(DDE(zV)_eR^P9}%;|!}qdY`yAQ5VJ7j!8_<|;-"
    "F$+sFGz?hzJoRgu3?etEYy45A*ap5k3o8I%jQFUCO(E)un1q`Ow~R1((4jy-+B*@lUZuswz?v`s8w03>cDK-"
    "F`F}0cDlPGHgiYOj=hfB$-$enkw3us!?CIHzbhU)?O?Y86{c-0@M_iXqy97(Kkfe)V9j;%KTt~p1QY-"
    "O00;m803iTi7_z*sBLD!xg8%>;0001QY-wv^Z)0e0ZZCE;Uw3J1X>es^X<;sKdEI?$bK6Lg;CKBBEMj90oTg|}mTb3X*n4VAp4N^"
    "ncUZD#FVtEDNE9WEL4XT@dbHPn=X|%jclVe4Q!XF%1|ZSy?wgC%Ot?v)Dl021E8kTN!|*g&h+&Y5MpVnJ$m%AW2Y<=lXDg9r$#x~"
    "0Diha99?X(j<XIttb(7^;lZiT5lvVIfRCQJs!S43KC=A1`twmKXgE(HSn{_4PILMZ(vTA~)D9R>jplQ9eC7)d-"
    "^;MqD)Yre(WuZP-LVa2n*}P0enluT+#!vG+scXPMw$YEc#VTnKu6p|;eB~F--"
    "6|_C)$@bmPNPz+m#e!Vse@vr9<7ohg$MBeRm$D#_qj-_VzdwmO-|f4RWippjT-"
    "o@;GQkAbn#Bio3hG&7M0r<Ntjge5~h@^4lm0~qWc?hS&16S>GoOTBFyX2D#@ywtQM-"
    "@ysQCQTxZKHPpYiB^Sa>}F|3*(996GVK)T5mnMnUsmFv}?UzB+|w4Txf7t*Q{tE!wMZialP4HBqIPv=Pvv*aElT#7g^lhkT?2+!U"
    "nd6v@pA5~RZ4TJhB+1Yy%FR~oi@0#C=I8SDPZyGPEWGN&_Iek5ATxh^_PS9fcb{GJXX+|@R=VhKJt6Ic2fIT200oCi-"
    "G9&sbSbE$>)Quc;YwIt^e?E>656%uwkIv#ZM=yeD5GJ#`BuT3T*7nh?xPc84y6uh+q1n#Pv&r6$`{Lcvo73ahC-"
    "8RSzIb{3{OIKLD1LQxc7V@tnCH*O+mq2a+<J9<5>xBvKfFEpbNu}E+mkaKU~kMoPLH0S0o3^9;MLJ7yokbsgK!vx2jA1b&*|U8aI"
    "j?o`SIZF?C1?*Qi;*LT&{rjQ5F6(&L;jT{pRnX`Rf-ij-"
    "MYNyo_HQzdS<Vh!rksEt;TCmaAOUBiyOs*4feNnbQo37(lzxyuJ>%P7hxF`0^+|ef#~Z<I__YR|XWA1)(m0GQWtu7lW-opS=Fd2@"
    "MwiaB%tqf%&i%z+VgD{UBUS_IAhO>3HYqlSz{7?oD=f_mbJ{$!ss3&mVvLcz5w^@i?74neWY#<k_>m$!vF$&gM^^q{+_iZ0A`xB+"
    "$+T0c<*Zx;Nha_S?ND^CydBJQlm@xAWbnPo7Sm2=RD#Hs77kCr^{7kDo5~c9Jw1KS_7y<HfUay7%Pi?Ab(u?U_9cFdy%TWHDY$V3"
    "~L4PrsedcjnKYJ>8wB<H^oq_u0<3v-u?55iqn!CL#sE<C&N|d-"
    "8N|cOv%o+yQ(l{yGDE+<I|zaQ61i5%A>8x35lUL3!ADa~s1htVPi%Om505jmv9M!G4TU!j$R_NZc9d4nT5+TT!I$i|$UKF=Ev*D`"
    "FUH1qwx|*BOGF)gTaxOZ7~(iE{yh;VP+W^=4icD7Kv!gRM75XK#*=-T_0M{qW}K^oQ3k4{?gkdWD;%To-"
    "8sQezE5YiB?jkiG?30jgt^R7r6uqVZ_FHw?z3-3k6X-W$NadUO1z;}d{-c=Y4T*Z=M4Fzy%##u<-x_O{*}oE*M>6`!6RoZ-gW-"
    "Pt-lIsWlSkQvW^ICyjL9Ck6{^lU7J!K;IRish={^SyCAhF9Mooc!tKL3{`UpT%bY^YrDx8A~D{(P){x2ho`nbrgR7^xr>!`t<qJ|"
    "M>jre}4Y-"
    "Z=XN?`{z&p@%hvL`uypCfBy7;a9;v@Jl+K^e)cVF#*?=%U&892oE`lWh_{26Z+Sb0*sw?n{Lgu@&hv}#BS=${*TK8p!>pd;qTZ1L"
    "h~BJ=CR+;nGT7%83BxxD=(jYuxe`TCE<ng<V0xga5=}-"
    "*3<#b|K+sMbDnS3$x&dunfq_!gVAHTQvh`oJZ4iMH_*oRwvuZ5{8mku|18Xz9sLT8s0C$d!x(bFU(jdJnk|k)MJipUA#KvZRwJzS"
    "*i~@pI;9?+nR_wD$f-g{HT3=fm*Lr7R5?5tuce_yo_rG9__L+~*vB5AHjYbz<vlL#VW)(3SQJ_d+mG*(S&4Z-"
    "WlYd@TtF<BNkFc?THbq%%SLMxiF0KVq725zC!brgy-"
    "!~;ZEPyZ9^CoDnL|`n>tVqSJp=DBmsf3FLcG(C`ix;G}iG1a{nhTz*jXS<e1(}gC8esmrO8gaUq@I_CB<m{w5{i&{b+AseMib-"
    "U5-hw+oV13zy#P)EHd=rsS^pup#7;q$f(%7#tg4YLO@iX33{YGdLQvB<OZBP(^{hqZ*WP4c8Gx+U$cps7Hm|^hyhJOAo}0(w77(-"
    "e0!Z}U>IHV;a@E9ImYTN@kq6Dp17PyYVhNMgBzmW}lOa)ZgZp{`9eDed1%C??-"
    "9K1~DgXvE<e}z@H2g0}Xf#=?<Zc+4&leWauC<vOWB+(|2V3e!L4j#mZD|@zU_B06pxg}pu>sd;UyIATef;zwZQdy62Z6NC4ROxO)"
    "izGvY>sl>fZ)Riz#~93V3eg;men0>>gEdG26=gTiN?6gJav(+K%DgOj&><B+-HVV4+~UHKgQh1f!9?AXe9YI80=SNO6zz9a{-"
    "f)I1X@!&)E{F1Rhro(m}p`{rUxR*({k!zta+{5@yUW_X8q7O$D4gYa77U-"
    "k?cYE@EdSXX<uKhc~*~#0l+tO9lYxQn3hPGPYS7M@Cu?g6;2Gve{Bg;BOXydIe&NB<&Hqo}djGR4^0ZS2kAx=$`TlbR+DDgM&v0-"
    "#<Ed{^;N^z>$b<B<|n~C~O>vP#Ex<vzr+6C_q|R2V?g@k4FJ0JwY}PeE?Q#yCMuLTY#hir$5gl)FUG>%m8e}1`|HP&K(0vLx5#6%"
    "{gL#*vFuw8`Wevn(wNq^I$<X1CMdnGjOYeLQ;1eWsRt45>8u~shT002QWos?6%W2T7W?W%ssziiXa6IB>tpn2No%{st3UzgDy|7k"
    "4*9#^>r8y7cfRyNs$YS@rFU4Wni(SB_c?+!KVw4MSn=@t6&auC*e&jR)d@d#YVPVAw<er9yZvurT#$e+?C35W3>lnm0?w$khkpZ1"
    "s_!+2eESvo_xtXJpxx@Ro2;UG#Iv<qva6|;6Ff=pM~4NicHWdSpCh4**wA&luIb7u8h=J+gA7OcAxgFJ~VM(WK!vYEJyN4`)L>?^"
    "SM|xDBWjwpeHr`0u0oEi;0}?Q(DtjuNEvmc<nU&1v&?!k?dqok|aoVnU}LDB$oPpXzf;D+Kxrho5FeZH5)|rqgSg0<=d!V2$}WTc"
    "OaM<kCHTvY+(GKfzs7rg{~{0<1{#rbg!W+z4h|}%QeIOC}UVMnGpvU55Q)q9w_ygK$@;VZhn4Yi(Avck`L%mwY@J_-"
    "i0S+utYn?abig5gU9VC5?q7?j0%x}Y7RXxAG&1KM>eVgW*T;)LTg8PWEmmlqW9u%nkUOy3Z7jfmis}3G7&$Xk1uesntDKTNIe@2l"
    "ord`lt=7`=Bp&*)EWtTc?Q1mdC(KLrLpNX2)+v@hQ3X8=kk~yQA=BZ)hUGUN7d#14VdD5vVY;CBJuAWyo2%nZrXu-"
    "8{(#1W&ee*N}xk?l@uh#QVe$&(v*=NGkb(Aqk@|RvkU72?6t5L=r=+5haepNwajRJBB=++gO3Kr)_`s6Qfkl%vi-D^F}|4+L-"
    ">m%Zc!2kuTO~*m^FYWF7Y-c+9EV$F+Px_3`POKFt0Hjk-"
    "%003VmqqRw9Do2BSDep%}*>U8(^sFaQoswj8GRIPPI%V<TZ=VXxX5|70O<z)>Sk=#dtTj4dH(Xf_1y-!sF`C2Of<E0S`bQYL-4-"
    "%^$2K-g-VOd@KG+VjIlqdF_GKvR4Un+(OCQdpu`F9o`TQBREAQd<Z|#*1AgtN?l{j`bMIIqD?!ns2g3ve~0{_<Z1*vZkpM4!4@MM"
    "Z|nvNh~aldkJQRtz|uwRG+L4PEbZT^aIY6??GsHizGABIgf&i5!p+?6mc9BYh52=a@3L9wdxEHlD^cED|D=R%MNh$$0l3Klb&CKg"
    ")cuhm6_cpjPJ1cA<1D7#hT7Q{zBX~sGutg2>MaWI#T*t?&~%rh9dWFvTm-"
    "}w1u6mgQ?m=w>B_L%0%0&Uf>wNFUp&u70?aVl+;{;fWpWI`z$lDK$2&~j+o69sWzL%?TTa$%&(P|l1%QHufPGii)8Kxm!B*$a6HQ"
    "jy{0t7z#OfOUUEYHgC6Sds*jT4%2_R{YeKK($S8bt8&LTQw=b3QuN_Pzip#7f6P|E&u-i98_@F+09JVY-joAnF>0^&3Q12y&XyJa"
    "nOB1$(D=O9zK6;;Af?M>6k_F*_7^a;907+AWCkbqT@(-iWD^Z|lm7^Y`t<t^q$gy$E=lX$|OCQ^#$=Iu|1|2bfWyN)CKnI-bkV;3"
    "7G8<Dx471j5;$rIK+oeE^w~Wb+t}c!uG0j`ak10PfEqqul;HIfdo~cKzj%G~1sU%}ZgD%2OyB-"
    "d$V2&+MNoCg~m!xbN)FY&o`rIOfrD7kDgw7TrXmWpGN=);>9l%~Llj@F9+E?pN8^R5c`Y}j^!|^3xISPFbiSIFt!U3N@fY^sIV8%"
    "6Wb5!K8;mcZhYU-"
    "x;qdXt<EJJ@`LyDVA3i>ZC1!fj5beG%#;J)ha?$DMO7GON?9KcFAK^xTf{y5UY?(E%^4Ge<>L?qYCs3St<Fw&O*?_y9G#s|^Q#~w"
    "Va5nHa8-jus6dG7vE@*gTu!>_W;$5-BVLQgZ^OXx`%L7Vlh3@yQ`b-"
    "H6!fjv&eM__SWXXzT6P>SBL+eUkcih)2R0#20x+_B~*Vs{cF#<2H8R?v0mY`X4JrilI>AghTcri^BkR^>`2Pb?bQ%#A|fm(Yj`a>"
    "$@b7GQ;0*)2?mW++-X@XDE-"
    "Uq+yLeQmw%sH~DSa3mBy*?=f5iD*$}NYmszDByA`D3Q=hDwSU(a%m=#X*4iRS2@0HantePFXSJWJYa-zOq(e%FZ1=X;22C4W+|3X"
    "5dYXQR<_GJis_>8)UPNCmg~CFsbk9F0vBZIP(rnR6J8b^8ow!G4p4B@t8+x?V$_s*2JA=;RU`fIE3pKsGNuYcwjPOZ@m+=mAE~Iq"
    "HrGh@nCBuXRQj1KbM{ejwVZ)h3>g-ks|0jG>wl>OELPc;FoIev6I@&)Qn;Wb;JP9ps;K5zQ>EAe$S7-"
    "ojO!YqcL^J;05J~Erz#l&5bRY0S7ddO@bf`C?Il6ukgF9N^Ps)B^@yIj8J{pjG*t6ZUfu{yw*mdSigiF<3f))|vM8%1EZEON6<9?"
    "`jeSPhrlm^BU^<!Vhj7D0eZ@v<8_QbRKH!2(jtgf5%tKUSv;e_?iNq@WDg7q;hy9;M@ZZ5dz!vF789hrLvnk(;0$$%cm<Ez^T_kW"
    "P40Zm`Ki&LPU*J9r?itCkUT8Q{)clERHWctt4GI=_kKA<Bgkk|oLt;HvoQqq`-^8-SxQ--O686JD6|<Si7;9|-"
    "lClZ@)y*8e!NT2iJ!f$zvdb%wY3mtm+AEl_94%0z>1{|gPIgH+Ryvu)k=!X$Z}-"
    "XnDylH5l6BSBfek@b3qqS&S>`SP)rKfVxnH%Mo#S-"
    "0bH|hgg}Pi3|8OOI0KfA_9)QQPOm3r&wu50XF)<MgV!87&(P7+h<I&ji3{+FeP{FjdMY^Zi-"
    "`<^W8yi|xAC8ect%6J#0(dF%dw}4=B1@syTfMNtLv9UcFm(y7rtmwT2uSL#mJAq)ct5(6egr0?u?qUN@NiZ~iN)ArVD+&}^4d;|R"
    "sTvgP?5I+j)~r`jNRuw9LcA4>vpy5ewzZV_C4_3m-aAy^|d@3+M}(tvHLp(xZKGJ@N{t`L2Xf;)nt1yFx}2TOjyW5prhJ%-"
    "(e}cak~JR+p3M{dN8wfmZwHQlSM0Ag5`;3&^gydsG&BqN54m~K_U$utdzv$KsJ{Kq_ROcfr9o2SKOhIp^2hPyX3~VCQ)m{PNUT_*"
    "5096A#JqhvYR6;q>*-"
    "&T!Rkz7xbtA;AL|awoF|;KeO)V_bN6+E}8lEALEMu44Y2P$D*%HsXu5@dj;$M^zBOID12M!K&5rINvX((%u(`IEYF;>nq5(rzH-"
    "@9Jk5eCc!hPdfuQV}U2g1EKYN8>c5z2pkaC~)#0$Li1h?Ka2iRrb_oe}5Znpx~$<WHs^#k|T3-"
    "sjy$>)|E&R9RV^)TaeE_3=^?W@!HPPvYnteKQ54hES=;cV3n|5_P-"
    "+>%^~Qm%k@VFd=FC;$NL&S|)A7TeFl0cL=&U=QUT8FWO3Dveo$4(W@V804-vt)yzEla3r4{#ot;f^gK6|7U7Ms7p6M;+zsa-nHm#"
    "&?1KYw@U7i{{b5emW=RottERmRx)i)?f>!g^~oVEM1Kz5{@aif{Xd5@k;mLb1y#Ic1@j=HMC^(S^ub!QkNsr{R<gb8wOj|wH<V-"
    ">k}cn}(!cS16ROWIXIKW$ljePP&n&-ea+DYgQGlaM6>{y)-"
    "*T?fnI}JHk%|0^r|OH0Z*23kqhWso=}U1swv$@D#14Ndekwu~9pohTPNB&flqRz>y+a2H%?0*H8Gz%)*nh6c{V>$?{b%EgVc?i&_"
    "MepAqhy6Cwf?pIaPKOI>nfM=e1~B8nCUgVzds=k5a_>3YCh0M7R*C0oUiJ1zTZBXb<q#$W9OSz713Kh^Zbx7{DMvL(8kDW+PEzB-"
    "k!Yx73RR`5NaD&JpC#N&dYV45{@%G{*ZiW-IIC*o2I4gLECSN_rqvX#1*M&%TbjhP-mBXV_CMslp2|>({shl%&J4fe-"
    "{i%Rw%G(ufVnjQ6EPbRyRo=(8}b&Pvt1@g;-"
    "6)MOM{~M6~1369#E`(Cy6zJ=Y%$^0;qC*c(G%d*h3~*uPvVjlTew=We4Q^I_NNKxu)>SQzqAu&$FCEWR4STOdGLB-"
    "Q*Xp5^5%ys&2_m*fI#Y2bg~1Rc|~-2>0k*K`|tmiJIG<}er!yB1)e4^wK8@|=(`VQzIo2>n^`3q-"
    "XIoB8<gg*q}MDvq&}^1BiNJfKtm92J}gd`!k1v7yR2Sf8wkV=Ne43?(u5{fUuV<5ss|`;nymRQ7XUUh8WTpn-RBf-"
    "5AFBFXRI=`_Tt#Wyny%U;6niRsMkG};*kyFkQca;u%pY3p>X=S^rlfF@OTOU3UFR-"
    "*+>oTA*o`OAo>FzVrwfgzcqwj>$*?5%)6=P@tBNuSQraRuYsPy_3L78Y8qc!yAsQVxa;g%WZKFDdTxWY;~uK$}kC=DHse#0Z=;l("
    "A^5dB)nmtkE%T)63B6oPk`pmIZ9iN-"
    "#Hgmsdmta%isyZV3$LaCVMG!>CHGMeaE{GfWt9xROp`Ut++Phw_XkMOkHnpb=rQsb1PqmY4H$gnMD_Yaf`k3g<*lTqX{(@#qY}i4"
    "mg?10=J0fx8NbYXKRIFw;S)H=IXt7dfxNIUf&#e+}G+-"
    "+}cu*i?a8`pX<5)N)bzJvRK%U@s+_$F>p2T@+=}wzsFGrB)1PO<CYsmi<1_f~5`=HtZ87y1bRrj(pl4I}NP8hY2GIkIxP$cv)d>v"
    "70N@{T9%3Y<z+7BI@NXbgS9adDeP=;dIAGUJq*J6BZwud90CKaw54{@G0J5P+=0Mk4U7$aSwqZlBywzAP8YFv>p%~hcl@&9|OSdE"
    "(!ATW*A&SAB4)4p^-D1zc;MOVmBliz);25d`j}(r0s8*S5@CsDXewL&`9U(qVVJe-lJkNuLG6E)5pAkl9!^QDuL+tTxQarGL`ooj"
    "kqq8e6QQd0{kyVg1Zk!6idQtq|ueQ<-*En-~^$KalnVTIp*meh$jp>j-"
    "dZzsR{z928$2$<hhBrUc<&sil)Z$QSy<k3@MBBM=}<xgN^12gz4ASzaS7|cdS&@VuG}phAsJvP`8iql7_^jr&WFLo=ShHwX7gAxT"
    "oi``U0ySV!9hD(z+BnD@|%?QJ|%(LNzQpI7%Y2j;QhUMFMWM&74vNPn~UnhuP&kW`)^XCi!YNb`!DKRV6WO>8%r5{x~{qVvj_W9>"
    "ID6<0UBP9d2jThV6ho2$}PXfzq!u<AFC*h3KZhdqMK$1a#?u6Y5<G&g>~xAMxIb+!Kq)ULSBcW&o+d2WDC2X?g_p(W~1hg7$0K=V"
    "h_T@ir>hKa#!5`FxFVG1tTU&m7I<J0L>WKI4%s9NH(mXwQsTQK&{3fZ@-"
    "sK(=Y6V^MNF?xmo&W^Gdb2FN;iX>9paF&M%Oiqsf$gS%#I7kJWxwJqguaO-"
    "n<ICpXOap~E@QrS^83XVS*3|MTi?A3?T7vc5pHa*+^K%<;v^NaoQZu&8#Arx!|!)Y2Sq2+J@I|dPh9Wb5c^%8E%*+_NcB5^Mhc_O"
    "x^G$8i3BVzbQJ|@#~ge9kb1Aae3GunP-"
    "@zBuU)7I00UD9ITnIJ36c}LhcCkzcmJ8v`rqt|czd;U)mlTPMKNBB`K@K_vuB`w%<)5!Bm;NCOJtJZhm@j6;Jj=&tRvRT&1ovMmS"
    "v<7UdHJvj+tGy;9rBJb)FiUZ0*0I{_$*V3Lj(1A+Xc(Q*m7H>4I`<{Z9-"
    "IPCNmtzUC3A_I4egPxn*t+6@2UELSNW|Mi(HT{AL8aq%VkV=`%G~uWMoAzfSfm;_m0e@hQaovn^<I~Os(s+Oedv=bDa^q_o5feiF"
    "Pk<G08-"
    "IcZ<~)_wIsjNsMI8zA2;~c)~(bPa2Ad`jDApC>MR;>L3&={NkKogZlZM{m_hbv9XW02oL?+g}^*`pIlOskuJ{pK>vOdOg>slhb8;"
    "a2qhpLNLujAncOxqLn(|T2`xL$nwYd`kzYTIhcMXYGx(Fd6UM_fNK3(?YE|1hD`Dfw`#?k2(Pj_YVmx~S!FQnUHg}s3$ihPlBw?+"
    "%1fwOxa#pLTni$bSqHpLawY2nSZgL{bVk+{YcY@7xt(TUOXY;iJNj469F#ATjTesKAC0C^uE*}g&y4AvnEG?-L09LfZ-"
    "sFFD$a@cI?WL~4R8QslVyu$uACUB_G_eG?&#k|3#k-6~Fq-m+u8o&|W>ac_0d_6zSY?IU$rFnO-"
    "WYQ&lno@+Sclg&G*Oj(KTcx3i<>Jjm#+XvS(C+zN(cMospEEakt>vx&63<G%v@<C%eUpb(3T+NFp$IMxef-"
    "FTiM9%z56sIf{M{(Bg=)zvxS-Us4HIf-L#V|_l{#hltCwDyIUjsnM)r?=yH*IZ9ID_Q$wZTUJ51GUWS;x<-"
    "gawp`DG+Z#EYludTbsI@=J7m<K$&nr`1EWo=)_D-A{-"
    "QJ42lWpgj8t=@CY)4}p$;G@<C)(5CP9|?^wKc05ndu;4k4CABYJe`ckpgMT7-"
    "cSMaG@K<lMt|^_gkx>k_hw2%RF8*qj$C>S<w|u<nX)7*xW34frYMV_MO8)&O=_GNvxgH$K+29JB3lQ&I|ZPk1F)hZggi1wc|=yhj"
    "O+Hqx(iJ?j1GGO!i=HAyivyr4$k+<)KM7YEh^4^GIo^O8AFDtzj|_39<JEt0o+NrThqP$%_u*e)sR~j?{5P>v^|Zl*`{B5abVBR1"
    "Z+G*mj@g$h@|Mt&(H}1-_BH)TkTkn_6osxj<rY%d2<5?ui3MP!>hCCwsz&}SZ0kk62_DIrKP{whV8ix`j`#|`Vv-a-"
    "4qDcVwWx9au*Du7)Yxh%qhN)IvB)-"
    "@4oVNy17tfUv5)%L3zzrQ`nfIH9FcGv`jmeMe+vpY0u^7+Sjy(aOP;_DkrC8zw>aJ%&Rh9Ou#l6_O)3|IF>th8f_;tSX^qUCk`_X"
    "HoEB8hU3F<LkQj#kFmNV@~bSct@{r++ec;0AsLiWGKN9N^TjsunI2?&zHZ9JLSJ5~Ljpw^n^>*o=buv%3mtc9pIsYxcS`#foK#L`"
    "ced1%8Ft_Xx4ABwG_&r0v0<Q_41Fge-~|Kr?u`Q2df9T>+xIzsmGk4oAv*;pAt_yQH;f=f@<2(afn_!{SRm3WG)A8B%y7-"
    "SI9AvF;g;F0%Ig7uE1COD+F%n4+F(FuSwinx4Ldff*IqX<X-"
    "LbqyYIBau#cClG(u?E;=X<2DeAU{VCT*1|FweimkCeJ1fBJY?Ha|f!;!%%L-"
    "n~`7Z|eXMM*Lm*__c?jg}xCqI2h9oP9*XItl|eYe)HFC?^Oo)sJL!pbUx*8g#2ZPu*E6&Ku6uHM+O(p)s1=Mnps`Gt-"
    "y)(Z#v&&p9J-"
    "(RZW@>+0}_3rg?yV6>?`Ic)Yov!;VGN+O5Wk)L{At{9*?`CXLe<S!M)(Sdop51^?pI#6MC&GpXlCv!VlCnLzh5ZUOwlU-"
    "~T8)_vKr(%_tcNoBMAa0~=yAU+2VbD>}toqf^um!Clw0FhU2Uyd6sjiSW;p@<7FH9}U>z+`I!Ao?Cc*#G?lnA9w3AhnR_B9+USVV"
    "FlU~Ez?Tg-"
    "559`GH}I>+#ftb<|DzO>o86wXxv!=T3}(DuX$AOkyK@Ah%)f@?lPV(M?C388mQR@7N40+GzG@MZ$R;b(1GxGH@^70z&`T{#)G*=h"
    "PCD-YqQcOlXaMc(#uIMe@-lcPCNbC?QqQ8P!w=9*t0(#GH`*Z~S7NG#jDOS3sV-"
    "4*X!c+BI7LbVQ#`tu|r&twBzVav$Vt}Ib&joP^)m9B_ocrc=+SJw^km)-"
    ";hy&kawwp`7kfKJwJ`<;B`P*H|ftg*eic$whnZ=kvFvHQHmiNQD4d@)VXbzw6K9ZU;aSt9yZ9-P1M3M`$q8XC(8^Y<2$De>mI*-"
    "WnX*0ARK;0Cb~VR&)@`HO4d`s*|DLIs-UIjD1a^AzA_;d5?{anaY$8nmmk6%%Md&?ITMy+3}?f<#1VnT+<#hHZRXP+TH+#2z%ltq"
    "6qSsNP6~l>JKXqoLyFHHdiIge{sXm?sQfl$c+2R;b%Rb9av+vMZj<XEQ`R>=41yT;rjg3Ta<fE4*BjChi)kW&Lls*7i0#Z;x<c*8"
    "X1k)*%Xz3;kxug$^>mlJ{Y2FOw#@#}8>^=6WW-"
    "R<P^`_qR9NEXv_BZ6}+!nMV*5bH7paO8$HwElux_`$bl_A^kGNG<=6gwy#<<410oNt}CaVj>1@Q{%_Iv7G3D7DC1{fX{CEr8T6v<"
    "?)7-"
    "USqqshY<5DA!us0T{yvYyw*IZB9vq+C&`Eys2ed79DseOoJ`U~8XTDg54I3WazrWL}3vpwFIIT67oL=uHU8S)18U1_*Xz#DqJWul"
    "Znr;N6Is<WwY>;6dTCJt|(HebSRL$2!nad98J}W9NHdlzH6i~abo>v;!((^@)+ZPM<ty8PW2!ocS5LCpi&tqy^xP+s#V_I~I6iUO"
    "c(KHBNCY1XN#vc<@{XUu+vd>SW_rLd=wNmOZ*A3lc%gJ-IsKbsFrTwcMjTb6b_bDkkhl-"
    "x;#nx?wd>@~=gl|y^@w4!6SR!?~kvGyo2I&lZQAn>dg!fhM=KXnTdYe9!kN#U{iy2EdG_39>ll(!AC4b*nE%3Em&g1NRR2&*a_dD"
    "qF+iVa&zlp!?Cle;^)O2RaL+}()Ai|OWx3Jle%;>iaaGAFG$SW7-{{OiF{vVbCydY7}EdOhl0Q8n?41RGXfPG-iMMa%h^UP-"
    "V$l8VkX&qDh<uwEPw2J%NZ+9`C1H_+X?6xP=6+vj6)Lmz>{EJ5Mr!OfyAyWHy?bIk8*F4EvDxGsS*46V7g{283o(@NSYc36i5B6N"
    ">;ly0}*>_Kp29NjCI2VJu^W9|K++6&fnQ09SwoC#U`;~Qv`o<N5q}q;pmhKIt*7-"
    "jNxeeKhQU?@!rkWoviJtKl)w#f*3&3C8;v2a_$K~>rAf82w&at@l!rBY65v|I_Xqn-"
    "8_dnXV3_O6EIx<rEKXWO6C(R+2zxU<f9URc5vz?&R-+a5bOE}cZ$TA_WO~sfO5PfnHwl*?dO}qI{?z=m<dn+Fwf;FdNpz>URW^v-"
    "i_RYaSCF}QG?Q7Z|x&gRr54ZQGb7}8JL_S0D^J{F*U+G@p7Cx<Of$6@$>RQMft*txQ{xdPZM~98L|NjC|O928D0~7!N00;m803iU"
    "VaTUi>000140000M00000000000001_fdBvi0B>w*YhrI>Xm4&WUtei%X>?y-"
    "E^v8JO928D0~7!N00;m803iUnxg3kZ1ONck3jhEb00000000000001_frbD80B>w*YhrI>Xm4&WWMOn+Uu<t-"
    "WNB_^E^v8JO928D0~7!N00;m803iUtLwQC03jhE-DF6T!00000000000001_fqw-"
    "60B>w*YhrI>Xm4&WZDn+FX=8IPaCuNm0Rj{Q6aWAK2mk;8ApoZLr_Oc*001}%000&M0000000000005+cuMq$MZ)|C6VsB$;Z*DJ"
    "bZ)9a`X>MmOaCuNm0Rj{Q6aWAK2mk;8Apo`yp#CQX0083=000;O0000000000005+cSrq^PZ)|C6VsB$;Z*DJhbz*I4b8~5LZZ2?"
    "nP)h*<6ay3h000O8001EXX+$mG$_fAg=Oq9D7XSbN0000000000q=B{?003`nX=`F{V`y(~FLpFvYhh<+Y-"
    "KKRc~DCM0u%!j000080000X0ALuhysjev0K$U+02%-Q00000000000HlGnB>(_#Y-"
    "wv^Z)0e0ZZCE;Uw3J1X>es^X<;sKc~DCQ1^@s600#gE0OkPz0Gmkw0000"
)

runtime_bytes = base64.b85decode(RUNTIME_ARCHIVE_B85.encode("ascii"))
if hashlib.sha256(runtime_bytes).hexdigest() != RUNTIME_ARCHIVE_SHA256:
    raise RuntimeError("Embedded Version 4-A runtime failed SHA-256 validation")
runtime_root = Path("/kaggle/working/olikbochon_v4a_runtime")
runtime_root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(runtime_bytes)) as runtime_zip:
    names = runtime_zip.namelist()
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise RuntimeError("Unsafe embedded runtime path")
    runtime_zip.extractall(runtime_root)
sys.path.insert(0, str(runtime_root))

# %% [markdown]
# ## Run the frozen reproduction
#
# The official labels are evaluated before the authenticated test is loaded. Final outputs are a
# validated candidate submission, honest grouped OOF probabilities, aligned test probabilities,
# and an aggregate-only run summary. These are Kaggle runtime artifacts and must not be committed.

# %%
from olikbochon.v4_kaggle import run_kaggle_v4a  # noqa: E402

RUN_SUMMARY = run_kaggle_v4a(Path("/kaggle/input"), Path("/kaggle/working"))

# %% [markdown]
# ## Expected outputs
#
# - `/kaggle/working/submission.csv` — validated candidate only; do not submit yet.
# - `/kaggle/working/v4a_oof_probabilities.csv` — honest grouped OOF probabilities.
# - `/kaggle/working/v4a_test_probabilities.csv` — ID-aligned label-1 probabilities.
# - `/kaggle/working/v4a_run_summary.json` — aggregate metrics and configuration only.
