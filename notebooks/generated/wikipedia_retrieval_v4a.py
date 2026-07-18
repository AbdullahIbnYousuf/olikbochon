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
# the official competition input and pinned `abyaadrafid/bnwiki` version 1. It reports the original
# same-OOF cutoff-tuning estimate separately from nested duplicate-aware grouped validation.
# No Kaggle submission is made by this notebook.

# %% [markdown]
# ## Input and safety contract
#
# Attach the official competition input and `abyaadrafid/bnwiki`. Keep internet off. The notebook
# authenticates official hashes and the complete Wikipedia path-and-content manifest before parsing.
# It never prints test text, IDs, probabilities, retrieved passages, or individual predictions.

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
RUNTIME_ARCHIVE_SHA256 = "00a261ec2981fe0f9ecb64e1607ad03201f2033c778bdbc88393b7b08c90c544"
RUNTIME_FILE_MANIFEST = {
    "olikbochon/__init__.py": "a77645398a46dfaa17cdc02963e9dc72c82f264a04c15fe5de11a0d94cea5cc6",
    "olikbochon/data_loading.py": "541e889021cee70c984152ed203ce95c0e835de58b67015798a25c3241a519da",
    "olikbochon/metrics.py": "30a8268b971a5c9d737660d0529f4983ef3f08e0d225fa06155ff2d6fd9e4175",
    "olikbochon/modeling.py": "0666af66702c0a2cb7438dae3925e79c1c897798271744ec4072035c2bc552b4",
    "olikbochon/submission.py": "fff5c7e2df7b305c87e043cbea123ba578dcc400836450ad18ca03edebf998af",
    "olikbochon/v4_kaggle.py": "a236f4d5c614d6f77f0f7afdef41ac206175f0cae3400ab5458325724445a5ca",
    "olikbochon/v4_wikipedia.py": "b4cef61abce6962ff1862c2e91190e6255ed1b3e6de7ac57bc0a58b5c5828437"
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
    "bkgxiRnXz-k@oIfAsN-hWU_0|XQR000O8001EXuG{zV-wFT#Q6&HX7XSbNZ)|C6VsB$;Z*DJkG+%3BXJ>3>E^v9RS#59Ix)uKJU%"
    "_c#WXw@}i?q1N;0t(-vl=%|ip0&`mIZ_?Q4VipN|%z7xJCZ^&LJs@5+!FHRupzjo)_}G@jRsGdDkpWj-pJIAr~wrzlPgw%19=ods"
    "dV(EO{nKl1WnDGZLjN6y%1<0!NRJ7DTdKW^olES>SozWRl2iM<`9IvXYEa!gqNlOA-"
    "o^l}cJMnV7NPi%isi=V@4inw{p)!}31mo0|Cx{L~XA-"
    "?4f$XELEtR*8~HGrP~Z*w*t4v2Rp~YM1XxSP+pn8bcAo0Q}G6daQVW9ZF$#5$sqgc~mqCC5yS*DyU7<FfSN=0&>u47>a7M<At)0;"
    "n`F?jcGeRBadOq<FI7bpy7TzrceBV=b$C5`TYrje`m|Z`s>w_e!M*Y`uXB|rUoysR)1ezEY9ia^2_<<KbNQUYPr5zEpK3G{po6X{"
    "ps@jRFhsUE>15$)9dwOy)+{(sS$g9vHJ35xuzdKEv^<H*UKw2{@d!;6)fOly}Ve{&x?!I+46c#uRkr`z5ifQoG#XjYhYiU4h){JJ"
    "}xh=m-"
    "O>;y}+>nlB?y}z~~JuV|969&77^y(H>tf*4E(X#XqRAmA&fq<r+46z518g{BO0Duh7Blkx4xs^CHUN7X?owJY|I*X5g|4J@$AU>X"
    "C$(Kp?`D?y{JrdN9qx7y@ILvy!VIDItTbsVqr&#DOObH!NkbHC2SWJhdajnrlJ=WXrrVYcjD&A^kSthNqyhUTzU4jMiJ(gV5bBlz"
    "Tnb1YedV7Vn;hvgAn^l|3@t!U)}TGMU6IAynzZT&8Fz=Bv*WOus2P`USsFlu(`!UT{&Ap@<CCjNm4wdLI8t24zE#?@)UD2ezM{xZ"
    "-)qc2iZ7fM7=OC*gvq$pC*9{`6&Bp`UW_3Mnfv_|(;Wdq7`D)!n!Xkp~=#(w0eAK1^t`=|TPkTwg+6aRsQES4H?>z!@r=X@(iuWL"
    "Y|$LSfg-"
    "2Oe^f3FgRGvuaR=ykKO#&)HJSO!|rURiJS?2Y`t#Vx0aRO0ZePG@w+3J2(Sgye3wjB*)5!CN`Wt*7_4w;%BwA6BQP>Y7Y=TcPftB"
    ")q*cJJdT+lbMoEeu{R^|S>RX6p3C}M4RrED57b3m{gl~L;k`_B`ojvtD&}QBgu90NZ{pZ2^FWApX^q@E?1MCWV$z=?SF>A}qxI4a"
    "itf1ywsw?4aZ4K~foO0MeEvzIPx_+t-"
    "c`Ieb1eO#EUNe2*|NKEThpW+FzeLXF)jifFX&Cb5rVlH#BiV&zLTuAgc)I?zy>5NBF^V$VOlWE2QXtfe+PjEXDbCm6!W!&P$hpM-"
    "oJ%6Yqu?;EQb7>dsUen{cL9HFqV~|52{hZBSEi>i@YkS%(7AkGsGr-"
    "`Q95%`9=|VQ!VK!lMf*N5WOKm4i$vml+ZSXj_ay|jtn#LiI?|TRl=dM-"
    "C`GnrJJDTVBffHo^Z(98CHNcGOTP+?HdXOPs<Wo!Fi{{@~6(igWW^SB_yx}@!FDL_h*O<nmuU4PRwx$XDP^Y&R4&++O%^5?}2*w9"
    "P<^U2@#3q)LT^0oWcg95SLvav&1@zq5X^r^oH2s{*8$(9B*wCpj`+1w}m#5;9X-"
    "bDUzl<S%LDs6WgAB{Nt{?lXgqD?(E&H%if;#E^}|L&)ResXdmsgF8>h0xc=zA5)VUShDpW+mC*L5wBX;^m>i4367$le9DX%Uoe%d"
    "?Ix2!@7?2AW+Ml{>su}y;lpRj8?goL_Z4rIVFzu{|8XmQ>hCitP@@u+VJcxFANjJi~o7z1VM(4cKVm3^(0X4%hdeh3n7)x2n4g_i"
    "6cvKyNCnRN}(LL>lm*iXB@G>YfF)i;UEAF#22EKMDL67X#sr+668lw`G)?2b<b;z9v?=4{$Mlz$xPls2~Dl}-"
    "$yBk<lBkSM_?x7EW;<lApm9w!0G_yv3tG3nB7aYSms>+v^T?|O1>WDOdtE3ytl_CmLKr}=_6Ns%ISZTwii8E)Ubxu1s;IH`)Kv1)"
    "|Ka(|{OL*Y`B^$d?OiL`=A_ah+z)3@xIsi1XA_)ueCSgWC-po`PZ?r?d7IBTW3jzrG8xw)jp-"
    "#T}Fdj370{@&EwA6!MHK#LSSY5296y9PZYEoOFrdKM1=5eY*4{&jHZVrOI){iaDbd|T`CM=rY7IOXnxffw&4H0qSac~-"
    "z;h6*+;E$Cfs6G@svuB8UfIguN#g_TqVnkvD7ITfPJ{<R}Q{oh&Ity`zZuVH>dOWK50aII|^~haNW>no5`G=L-"
    "uvoE<6oM|)ZeY4fTw^=@d=;AewsB|OYCUbNpX*-"
    "I?$gK{;=#G%E+mMGN6+ya?jI)nTAJ>gpLG*${5*OuFz$5Bj}t3ZAQcK$YR~(z%yQi^&PELDCdQ!gMi=IfUuC?$sKMOaAF{Q(+}7B"
    "mE2%9%sA$LyrZ7N|DLY}U=rdyl`DISt2JfdqSSWbj1hg)IcIE3|SNg4KW{$R*TReN-"
    "umla)%TW>S?=dU46s={gF=NV}oFOAB<OrKlhbo#7>{AK;2f#^p$NH8;AHW7Y+MYl?(BwiX-BhNv``%vpoT%#`#j9_W44PIW2$#R{"
    "-0v>OTxB7#S^`q}Z)fBu3v`d@4j_^Nt%g-v`igf(-"
    "qtN~_x8FyVutRv{k|g09C*Pq_Z0qAr>=dwVx?z&0Wr5~zxSxGks!A{;J4aU4-jZsVH4y<aN?^_B6W@wB$5n)-"
    "d~BxO6~Cyes=h?N=B^i6G0DR-<amr2D&T?5JVyvc^iEA`Tgr$HPCNXdu;JIt#pl;(9zurZE>j-"
    "?Nh3j#HMPX3J~><Ot46ZbxN`XCWr6O9b#2~2k+<|(sr}sx9|;<F%X!Ri9*>gL^*xt;k*AvN*8brUh%i~voYo14Sy?EH=59^3!b3a"
    "i~I21`wunW!S{?tNFK@peH)5QK%1WS(6I)Ffr%e$u5VoyLC*!r!v*UzyK_Be$$r-`W{7)Waui`yGIf-"
    "@lRfsT2shwkOv1>zUZ`W&$T8D}9mX$KM1wODzOD4Dq8ruF_{G_dV)mnB-"
    "A60S7*4ZFn^J*=WBhr`@<yI!`w<{|@*}TnjmSdq9MA@Y05q}S4~a_|drxb)FwY@ApQ(X5MID^(YgWT!NU>H|JAoX7PlT=I(-"
    "00pyaDWN61sw6PciYxWhQn^xw+@3d=C{8{Z~KQ2706}8-"
    "Bl2R^7Dj?rBX#sgq>*9*a`r15`x>)+(ubrUhE_U^~oUZ>WEH0HuKMbexS}9$5T6=s(Qfh5i<|h2ULtT9Ez+albY$R5-"
    "~W@eq3L^1vZ=Ad%cQyBrtOA?t}?&k*>uEhtcs9uBN%kle*th>=w#e=c8>I7B@h5~fie?42+Shi485=r>NPz;=f$?xvw}Bcg3uF~q"
    "waY=A9S|7NU*81bS4k*xu6Lo5!&Fg%0MV@QCtthNZ<n%6jkq+eNP%ZIN?9e5j<xqc;ame-Rp4y{~)b}UH9E4~950>`k3RTb*IeUP"
    "L2Swu6v*lq109EeXvhAhB$D%}+iy@T4qnwQXn*S##(FF$RQoK+=0WgNKswcKXi`daAg6`$Ex#g@Nb^^?Un>Ld4p<$umi#s5S3*jf"
    "kC1O@o0`&v9&)$5^q+lrs&)!evi@*hx30|XQR000O8001EXq2Qi{f+GL`Xn_C#8UO$QZ)|C6VsB$;Z*DJkG+%dVYiV$0WNBe8aCz"
    "N5+jiT?k?;Bn1P@tYhN4MXH#6a6Il7FMv1K1icCyRS;($a`LJR^707}*<`knRO_m})BTUC8;fD&y_Hhb1H$0mX9>gww1x_8s}{j+"
    "$+I$pwRR?gErt?G2@{ULpyE?APrn<cBulwHS}H;F5jr8)DKb(*Dh$|`SGl-"
    "@g5R%wxY+nb}V@B14Yv$B|bQ8ZiD%aTQrm(CYOS$lDw7j;}i(`sWwJiCgkt1O+!uYanFTz)Q@{Itx|X_2rbu44`xKTWf^ssIDgMm"
    "^#!7IBSm<=fZrmA|NO7ioSepO5leg-X7hFK)fK^74gzw21Qr9>D*Lgm<srXDlxB?u^AWIrgD0<0;OmTf=7w_k5Nnvv+J-"
    "7iIc0EA75W!nlm)Fr`d(cu`yu-QTdwl2u4fyU!dKVOn(;aa!J_6_fp@MFr5JDxIfUT&DG{(+$UnVAUAmD0`g&(sepZS@J_!EEhi>"
    "7g^FVp3(yw(xPOGvX~-nntW$95~xhhrf~+d<UP8$6j4^hiP7>9p1q5+G@<oBD$Anmc-"
    "2+BwX++|(hS(|n!jaH7Eb`*B$}1+oC%O(`fAp=(17Wbphe<s#{(pjlx7-"
    "Di!6&56^m{Fdq6}0s+N;^O7xYm)VMXPYBA`>#ve|8Jc$lR=cBWu^XSddu{ZSmcyb%ZNg2c1KAGe<ut9vg-N_*|+uGXi?`+vG-"
    "W|O;J9%{qZ~OL(7bgctr)Nje%cJuVKEq)S4tksYZco5IK6!Bz9lUybdJgY)djjzJtMk!|=<MVlkMQZ%PPD(X6Fqy{<8KdMot__^o"
    "<}c7rzgipXXnw`^AQjlng$yl{PDN<wst3b$#!qDH``+Iv;Ak!_TuT@-"
    "u`xPYX|1IHQm|W>22*jOP+2&o$j)DvNi2blb!9ISwBghZTlPH#;fDwlY<k$0x>)ty+llWT>6UDUKP(58LPUupZ$$<#KCNar1qd)c"
    "UoQh8)u`JuVIpBZ@+(ea)xw@u-zG=>vKS!UN-"
    "8|@x+VBc@b{>c>3xOr<`ifM`zCo%pJ|Tf$+fdXZ@Y+9@~Sd?)Kw&d#As(y%SF+yOW({I(_==>Go`Y_B845PIo48yuZKGpKSM&$#i"
    "!&iMO^VTl;>8KwA?8u*qa^r?>s=+0O2CcNX`0Y&&^2-"
    "QL^X>+dr5bbB)0o=*FF@!r$D+0Is+#Jyc0d2hDgOLlhmCi{H>cE#*rfN2kwZPuIhw_u#<-m~d+Yr4O`w>?dI{jJ&d{?@a}w4ZD-"
    "7@Ea>mH^=1g!T7#_jb1XY-h(Fz@_3ZGr;}E@zLn~?Hgp@7jIvlo@xG@)*m9+s};*@iOEe-CQ)(CO4zj#%ArEO0l7Q@4FgCnaKp01"
    "e$n0u^hzW<rg;QoEkNxs`8q{#lM2K}ekq^HHc`evd|bt4CErYo9EH2}BHVa$bpGb#=p8WB`SUkNXU|`~IK(N|OQ2`~Vl__!kS$9P"
    "ZCfGfN&4ocIVhNJT*mn&3wqt&PRHwYxBK{aZzqI(_2%S<lT(0tc=Y<ktA98;j9LbQaeCdYosBo6)5BLUqqFnTIc}`&t&OwOlh?07"
    "vK%}gy%`;xgW!SZ`#m8UUXK14iB-YpJH4m}uf89h{_tWH9m2rp(K*08doeoa@(M`QoyYG%z{YtM_@6%h^QVs=KYjd{PapsF)5m}N"
    "^zq+6ef*D4AOG{y$N&2D@xO6j0((5&1}@%z2AlEp?TZ&&8vPN3;ONC$z8!sRn8!K(GtQS;cH!TH%q6Mrz1u!at0^w(Evbp%%`&gk"
    "IioM(fUk(}zllK$Cf>~z%e`U-"
    "svrdeWm?P^tWHVQ0l`7j=g>x(GC=#qvIc!#f(esXVD)f&W#jLa^$>uX_?hLy^K!{Th0!r+s7epbsv^4vz^&tgM)cxk4S3b4pIXM}"
    "gCZ~qAOiBH`GA`#_yPr_@wKr{jd#Y_QCSpbw;MU|>KD_iW&XYzaIPI=gO1njb}t?!ikj3xQ7)F6IIm%Qz?}0U-"
    "z<xp&5T_$qz|?MW&(P6OMG7!@GuANT25;Zh~;T3GtCqBLDMKMfsROiu)(@$MjVrhCbAUEa?1D|K;Z+Q44`*S?-"
    "E8PV}!<*`>tev0()r1%LFJv3l}ZR%zXJ6MaYyIEt9lX#28(IHFt@VR!}!*F_>z%-"
    "VCh9>NnmccJk5$xdkk*vO==dF$#&oL!qMyK~1AHQL6>i6N{x^JClKB0J2^qi_!Z^zXEgd5-"
    "lEjt{<}xfLNr703>>E^a2}jzNjN55%t^0$b%;32$=UWpTlGoiG)!IETKsh-{20NL5GUAQ<3v;Uab0i3s!o-"
    "V46JCT#|<W4RRLE(=xv8c>42&L9}aSl1A7+n%u%xxsgy{T1H!%1}Cr@2dz!s4E{0!S7=MK%i95ddXzSAB=du4S!S9z=fz?ZC$Be0"
    "v8+La-~-"
    "?hAQ~{roXv~!7WQ;~1#i8qxV%Ke+h(3BPZuC2R`8B;A5z?BnpBUAMnyl4xt0U3$`sIuvrRD0uZo1$@e1Ywh97Yp;0~YDIZz3_s|="
    ")rc>C(rF>=`?o(Lb)5SIdG#9<CRB0o(9964hfz}8-"
    ")Nm0xqYb0yxW=o4V+S)`h?R!H80O=Ao^T0t|a9M*g7wnJ>sCCVv9AsD(gx=<NXa^&L7o2qTVxS(P3TAZ>3KlQX#hIlaaNPqxE8ss"
    "=qV(Yj|0-0ZY@ehhxuLhb%V>n{N8i)m2WDe6cz3zL4H?Y*J3iR`lRE|F{t3<Fj)e4}x0T%cGz@*mA-G-B@AblP!v-"
    "5Ci3xLb$U*gbeki0ec+@#>4AeXN<Le`^Uk)QOT;3d^N%HPUxdxmWkv5NzIFu_vZ@In;v_7OIaJ3(&3dRsBWzkh+dj|NBF9d$T2?M"
    "BZd6^ZHz^5U;@k1}2(fcmgndsdD7IKaza%c-"
    "tGxfMG2(_>a=Ia7n`wu~A4Mr&46DSzMf%dDR_Yl$aQVU88t^z<{*wVzR5P>*;=L7w!<(&F;QL_gj{1a8_&#Z~EhO0f-"
    "RdB^O>L=FDuvbv=<aIQU^K^#li^mX*1?_mNY#P4X>Qc$5=W%sK#s}C}j0W5UF|{ei@Z=cEVgY;7`!T+}1eZ)Yy4NE1)h8)M_Ctzl"
    "6_!?nwz>jCV+#w)2#(_Dh9)`qvI2}pxkGj0!y0?n2(Hi?h4-G<eR3y(-"
    "T##PU6!Mf6oeMx1*9wXfqx3*4l=FU$hIxl+fZaX|9J_rgs4uYhBrrr3iK6}-"
    "0XV|`cD{;4Hu}S;FGin70EdTfn3%e$I&^x6u;(0ExAKe8y$(J*WrK9@=MTehV_o5<l+f<46*K@lTjBn=w)L2q@WQ%r%kVrS%hFKC"
    "KW5uUl^{z$Qo<O$O?M}NA9H+ah^x#X36}Pi3;2{)dC2f2Iqt3Ht>84#LSl>*t>fOvHOmm!=3zeuVGni$R{s_cOqt>`R*Z>bJ<Vu-"
    "xPK?8Jj}EVuPlJwKS$F-"
    "5dH9)<iW%Q4e1NN2BT?N&o5Qz|276P#TNi1}#hAqoFqrRG*H`(CT@}m4f<i3KKP6c<~5q!Cf{e`ItcJt}s*n{KDKyy20nM+{`~M)"
    "V{xSL#_8jbiRu~tD&S2j=UBTiZfx?ESSqdmis;Mvb3tl53&a`!I)2N7N^;=L=K^D!UmHsA19TjL|xw6yzp6apt(>A7W*(|3ow+=h"
    "_slUfG74)au(eftp1$rE(H+u3B@mh@P6Rk)wc^4K>x5Ckr@$1_kL)zFkqPGOJ=gYp31tSsIe{*wU(J0Q_U*&g)?(xmTxHmn?oMe1"
    ";DO`bX42U;WOX+*7JYk`Q1MiX&y*KL(9stx`f_$UjH+t^~-"
    "{^XPlE)YzLZ}lALzZO^kWEWuCZl+e8;QI1tu`prN)rK@K64+qkz&ZL7_=chQjFM!4!B2~MK9DYeO)GdD5$-"
    "qaYk>D|5MCovGQQ(!Ap+{oKF36?MC41M!pMcTpfpF{}WRs|BEBkZ|MM}<P2gp9YE02>6N*^_2QBGe88In27Lm0>WN*$5E1mZiYL&"
    "^?!+;Y^L=pd|aGm3NA4Wzo<3t$Yvb*&NSaqA?!@7b7s2fK$Xa#PqtZvjL~Y6(b~Dqpmii`_`T_bmCWUxX;1jOSfCQI(Ijm8~5#aL"
    "8Rym<j>iMI!J3)8bG@6eO?*fD7Um4kT{e!vtLE5m0w|OVAHo68NIyNd!H9K)=1o%UXnpSXLXDY6;L?4mDYYr?1*V(&=u0sS9a#mY"
    "b5#vlLJj@Yf27KT?BJKy8L9FfSFd52}UKk8o{+uOHQcYsiA&V-Zw4%hWpiK#Y@~i*L-"
    "?qVd7{y!qviVpowrNKY=ao8u9Df=JLJZ5GCBNw`sy=d<95BE{*@>J$cSgC<o$)#4ydw3P_p?yi;HUlz$jiLM;!B1W&|PS)F#V^`7"
    "<*X<G_B$hBIrfrP9dKA^6`mc&ZOa6ni1DZ;zJ$wDhJZM~{?9%$%d)}|^t+RlW<Z|h>I4BfnuOB(Vgx`hj?0bDn=2{Cl&($b7mYRL"
    "cL4Vr-"
    "w(yvw5rV3Z%RTrl#KR4!L=)*_Emc`EmO)xX21=TX%0_^!bE^j$X1G#j>y>@8XU}7D=5`hBv7Xt+F4~4<w3LP61cE15J4a2~sE8Lc"
    "-cwvJUAOgzb#$aF$@=i_8cbCMcM1vrl`_fP{dQE(Lu`B@gtM={|NqJfXU_5Odz{t=+8`R0}IKuL8?M)FD3M?X$YlIv>2e{*R!3Yo"
    "eGaSUdU=>tOnzcmTlB*QWtfk_S7toVDTibwFl8s<hDyOq3Q}n?uQ6Q$*X|jYSR862+WG!~5>Lv^;Ca|V}JC^4J+`=!)6uJ+6uJ|f"
    "DCf{D070}-"
    "yvInap$^ZmOSu8|`$e@wQkhp)phDKC7M212<1M9{};bG1<MJd2D{5(VObp)DL*T&nHQZYdTOPJu3HHgv>UItY<G)<oD1zgS;tWLF"
    ">R5FQ5<Wfu|a)n^>Ei!!B;HHzqujC(_JYd}LR0*fRw8)n8oQM7cKTRl6h5TcN(73(RQXLrhXKwM6H(yq@%8yg@7`Px!Is&R`^Ydl"
    "Fq4Ap}W{3ioUX2l<i*8+DlqaBuvXOfDg;)YrIi?arZj}+=;=2@UQxaDCO|B8_G0j+<%N#o|A{wBMX&BirIb_UWP$`6}(EYp0!@8P"
    "H0mG{p$hjITC?yl{`WKIvKmk!%ImPlX$qqn9QLdz2!T`NC*kCz`UU2GUbrXQ#-"
    "YRfKTIDf+9$E<vnxX)WLoOEFNCu6=TX*THodNQF$}x%N-"
    "K@9)XC_4Ywbd12s7N&ygv^U_4h!}(leJ(0Qe(hTHfbqFq?|dGI^l*1R@FtxZ7hm^R{^*3x?%(jHJ7YA1HlkySy1{vCEo_W8~oITf"
    "5YFw7HLO0RB6n*c+YZp{or5<2of2F1MY;O#{cxw%}><@?nCc^kt~aYh9l+5KT%<d1m3L{S&F+yY&vQ}KDN}5SUMJE>;s4u5GK54x"
    "(WnW680TW*4^pJXv1m-"
    "lCt*xLe&+9fwd@$WH}XCr1bI%WZH59oAwGOEJpL>XlffGi&{1AumHC@ha`5&(Aj<BUr7~?s$gCDH8df}!bWH_DT>SnAlne7D7P$^"
    "vvr)7cJ`Q}_EA+<;vZZI7r?Kbkwfqg=JAK1rES>p`a0Ht!6AEIR%{lF77Zg_HWduz4I5jeeVVJ=+tY1gL#yh-"
    "(UPZ8I|@SpF9mK75L{Rf%G7$x7iJZMRc`7HZ9>Z_{Msb~lDe%W2W$tp+tBT=MxfvA$zWOo4{LR#Sd0t?mLJ<BujyzQrMgt;6*wzk"
    "nbplA-"
    "T~jkfp}`RZWi_qniOcXUjg5KX%4fh6qwJ3_Gn{m%>GubFz?jJkGLF3P#aX|YO=W)X)b~2aEpOJOJVZBVkxt6vsRh6l^f6YU?$5nO"
    "SFI{i&nHWGhkXsuV;^bk6^u67&=%Eib3~uDhx<zgYYUgv_E)RAsQKqD5`!<Y<y!9l``xUS`B0E9hw!wMmrGQEMXyxw5#|UbjY94q"
    "XK}JwM-NlrmmWwUia~XicOPCWLMqCxZ*#<rjzqA=qu8;k6P4DZM{2vvuHZ--"
    "{vX+XdKp(D)KRNl)M!)V~NGQZ2^%7Yl6AYh8lYs=S5CBdRi=`8)x_*$PkkPTzQ!yhGwZ*?xFh2+AKBsg{*L<C44j_V5+%BsZCobB"
    "*dIfU5y7@7nu4qO7KP_BW)@M{u*%K|FYl?epzsb0-"
    "T#?;p@^?Mdm$u;%%+vQN4Dn_7&Vs;un3I4O6TvQH)E03n8_%35oK{Na4%cZ#+Rk4@9B3`fhTK8d)QbvWt_iSiqe966ir?RKT@Ee_"
    "78q_x%ua@>d{iGB<e*u;~CLz;1hwe+>5sn_BAG{G)3ysQHS-oSVCse;UOJ^3Ni^#r+PWV{oX8pDU%wnplWh1Zw;Hvsb5wL`_=BcK"
    "C|^-"
    "<qb)S~!at8y=`YwnH96OD^;)2b8SQ+EaK7JdsCePy))w=eS4wuyxJ+pYoq_9~ppLpjL$Hw8j&WlOnlA5r>M9ds@7+WuVJH%}R;Y("
    "8h!P-bKf=3=T~$+D5Xxq7~v@>r?#WFUw3sk}aa(V@{LK>b`_1^rn9$-#^e^)U-"
    "nnoSW=09yE_YU9>~FZ#)(CL!!XYRsVuE7}O_4JRHl3anw5+rwa~A<TAOjBEdEWQ3Z_{DAZ>!VBe8XG8Ch{XKXR_XK7j0LguxcRYA"
    "9t54ySDpkwtxBagJ~VsCWjcX}7AqL~O1cv|7lz=*s3LLtwO2no*~4-"
    "k3FDxPFaLHF$`0J<zLr&rM=D<=MhIU}(g7npIUs+^W7n#PV}w<wx*9JSVw5}~7{N~diNLUou;K^7-"
    "Gn2z1avmI!pgU8Ft&GX~Ulfz?q*aK`L3LMf|4zQZw0iDg}p^mA?4>;(<5maUaOO)186o8JLbp%ZZ?)=DBvs2;NzAKhsDEiqiuhn6"
    "8Xy6=gx1~g!$Js4B9r`%6=w^bkrc2mB5uFVf23s9(8)!L?KPbz2*f^c2Js}$pph=m2prTC+tL_XYPN6$s{&~Qs&|SeSp(dH6wjdd"
    "IL>gXB>j53YNu8@{DHo&$#?czA%2@DCLXj6S81faC;SJ#>IkUUldeZ7}g~ZJ+ViF*_aYcoE7(sLN;GdVQyltVErq!4Oxv-"
    "T=1tesg8=NZ!0s`4F*8_J3dZ;|*z@<aC1lKBaESCwdjYlV%PQhMc6p9b!=u1gisfkdh7;LJRwwq?fbc}E>jP30LGge_t<ius-"
    "kpn(DhhPQ1(AWSZ(`tsh3W#d}3A-4+rF;U;Bfkx-*WPcvUg-VBvmbs3irZP82SlV%&7h8dDe;)*l#H?_W|;UwusLiSh?6XWvSG*S"
    "uBBG=HU~70flh-"
    "}q6KrV6bo4;OjIo#<x=>mAnX(xdk>@F0v;A7X?SUFY_XdwEOIe`j<N9t`WV#9Ug$=%q4lis{=({xkDMOV$|Wp5)bm&(xx_>=Hsj|"
    "nK}TR@dKHmKjpH5y>obu<5<w6`*fAau9E&rlGZKJzw@Hu{Hy!T^`XE$ZMiyA3xqHKyEOtYZ0SsBW$<Mbvn2Gu;CX3`ZSvO}~1=Cf"
    "Dh`>KRKJYt6Iw_Q?33d7lD0#_BN___3@Z9<8JdksA2hHve{TZH^AtZSBP#>RZIV{5?YuY&A$8UKc$3DtT7_=N7{(T{{oKy_L50+<"
    "{4kBH{#*Op3!df%R;u&clF4CWf0Fw%xT&vTz!!JwfBM@SDEOt_2g0z`}E%=L2HxFQvhQv&cQL1a7Cw{E8D2vdzCu5OP##C-"
    "ZH~0ifmqKT$NewLuw6s;If`!$QC5gx?pvJ@F7`!TzIfV)yTH6ATA&YrTa=o{7M#*e!XP2<6jLe$STSe*Jaa1hW9Em91h4li)i&4&"
    "7+|E(!G^3CpWX2bv)UPz-(3z=3bY0*pLGokebX?quw9FY#S4dXx^1T(<r)-"
    "eD?r=EH078R@_Q_KL>fR~xC4#QjvIj*z%ka)1*gt~3is^J&Qo+02e`*_h{PezW+h=@a1BY>c;8aS}p6PN$Ase9&&Oh4%*`$>UJ&5"
    "&qkb=&dHA(qbK-"
    "R)bVN3OwXb5vqq{cWmw4D}qqzBiw6qLrT&ud9}7i%9Ejx8*d9i`ZD{4fl;*x<5PMT=+t_4Xz`+q|Pu#@PH~(A$=$nI&uvh6kOgj*"
    "mwKxML7-"
    "upy_jxER3BqiLzGT_oO12PBBCDYt?B%>W6$5s%4q>|*Vk+kn6Cq8V+zGI*%t?rGy`Xy)n7)9#`;mG21i%zH;uwDm3uFnZm_-"
    "|&C(h;%Z)E`mR*7@nJ<ucQT6+!1k5GTeJwc~$DBGQN(+T@f(Ht8|jqVyDUsxzd1jxul|GwAw2|QV12p2{ROjVjZKsn!M~%OfVQ=#"
    "lmSAm8KG$@=%WI1(!WI1wJKJMzktJMbxZmkF?$A7D##z<p;cQZ8@7|jC6SiH(yfBBf6Dkh)W>?ENTJ7yk|>rb@Ze<-"
    "e$j@SY)OQjq8ncC#i-r6`_CdiVnkxwy)jL!F_e>hI2k|Mt9i-qGniE2ZoNY5Y!WfBBI{m%+V2x9@;tx#R`8hCfJ~Uer-"
    "O~BVDZR<IVg-_g)w<58lU@lp&xiPwwdNZ@s>GERRd}juuKlI*_#BmveHP$aI7-"
    "5+pS2IAdbMqD6ka>vdqT^=I&dxXr`C)=LWTc<G#vDA=Qvp18Hqw;gTnK^u%`PQd#P)ZO}S^N1`wv_N9673W~Iq*$Y8REiQK8c3`f"
    "dPpq|{h8^|IA<{#c~LvT<hjO6!^kuFT7o1ShkGysE!~aV!^D!y{ISgk;k{h~iOABDN&#R*E6h#)TZ_C8kk(x43QYAxtS`nYF<^O1"
    "q?h$p1-N~7QrQ;oA{xQb6zBO&ywo$3QX$6Jm9*n3%U4bwo6Rsac+I2@B-"
    "B`o*VWNcm4O>4G2TV>6`0FcfTJih;fhKH`@|8&X32ytl%&lP-"
    "$=}CX(Vb=#XH}WAmlKR!{)dS8kSwI#oIf_&;^2$(L^Kjplv$Sv+lOV%f1`7lI6j142UA=ByD$NWH)i;0`YAwlCQOAPqjCw6zuDK#"
    "M+Axv$OmUnm4qw(fLiMtmU<FYgB6+ViEm-"
    "&#t07Qb}2xm!=AXk&mcq7Kfs_6V;aQdCb$o@*;FmYXVyZs5u`2jbBkaY`L9Q+p`$PN5^^C@AW`+@XdNdRhdJ75@#6w!P;-"
    "j+A!}e6o#l859b`%^yr9{YMrvYD1h54SzPBu{xd6!fI|}+C&Chjz9k?<M-"
    "q{ZQ=zQ_P|^WdQ4&I&*QqFBMJkx%y4g2wv62p>#h!pL$IxQlpk)Qa@qkPni7_~1p>+|18Kt(y5Mk;so}7`6%QyJ|_9X1BY2VwWm7"
    "mILh%JjZZvh{gp2nAK)6c!~ZN<(6Y#c+E4;XR~alR@)LuJvfohdE1%Ca8K6@u|BYmpT4`UVbOkY)^r7Xi_o+S1iAD}C&dFrKtpTD"
    "qI9)0|tlis@jW&tbL39dGjL9LJWhxeEr745ZW#<`mxtEeztVyx9%r<>+*M`+&IkOI;&f@pJ4Zrf7|Jcfy8g$7PYYr+T>JifZL+8b"
    "erf)N++4r$oQ8KaZzn5zYEw8+2B+ne}-"
    "rw`Volj3=<T)KX0xX6&tX(X0u_g<*#foXhzldAr{iSzsFvAFwp*|8?a*$|w=TptHPe6ZuRHvN>JW#cZao9aJHKyohwH*6{Pk$dSB"
    ";zj+`hbnf(ZEjXc^^zLk^Cp~Ov2e)~#CuwHY{bJ2PHyQd?M8F9K>^v9+F!i$GvNsDYzsUKq;*hO^laLfHxgACjA~}@O$*e!Du}-"
    "8>_=h~>nBj_dajdrcn+>yD)(isxTQaXQX}xtUXpI3C==;tYd^0wx)?PKyX-LDid#HrFvx=9DG=gu~;;wz-DC(w$;Le-"
    "T|4Rku*9lL>1g&Lj&2lxr#gW0nGx@n$F6ML7i;`qCvOajQ=*~ep1Y_&8k6HI^)c1qU+EUTxiwOcu`BMfhD1)Md2JNcPQFn%lbB5C"
    ";Z$e&BF)g{57(yl@Vx`&bFpKAt1aDHf8Q@=Ij=;sLb0T=v1YR6pIQPDyP36d8y$9+gRsTzg>>KA8s%f#{0OfJkywJxPsV<Hx%FRB"
    "2raELU!|ICbtuq~R+EF_rh{6!jXxz#!ri%5I5{eSG$ckGGU|0~>QnqafYSz%`$Y(}bqp#V5))3m;V(SjpbRg6f@+N%kXzhimMR7L"
    "_iZOVJP7z=7dyx{Mv?&4C0?E7&U;&Fr4g`!%s%3*S95)a6?Y=6<@Kx49$7^1<Yuq4ZtALKT!Y5Gn#3>*HJ7DL-)5g`ryxd#Y-"
    "v|>z?V2>N(u8>|o?hW`YsSOR%CfLk`kE@7!x^^aWaMV2>35Ahgr(jYOIj3p)63zR{>Pjg#fgf;WSEPZSsFG|{Q8(S24BGpP-sD7*"
    "ye4T&En~{c;C!n9!nI;b+FW*BN6#b*02>ev^;Ig5;fMSnJbd%ib#Y916q1{-"
    "3kA)GJ!#_yIcVqu4bM?C*z*<Rz9+*C_*d7*iN~$NN}ufpt$d;{k*}6-nYhlF-"
    "_2EAwCKn=h{a4Tbu^Ia0)D~wd!cg2=n&_lS%PryV;yv&8=a~_0bJtBEsQ`1>|3B1J_-jt`jOyG|xbti<@r%Hv^xsF~-"
    "HJe#W3}osF141A-z+v+cw2g9aobO2cF{XVz)r+kj#dxh3|X5N<>uG)JwBL<re0<vtoJc3y*sw@KKdse*Yz=YCAiuPQ6lZlKw_M-"
    "$l=Px`YSBI>k=U}3KDp{)vOS5~WgGA)xPY8$Cx{ja&!<~AEQN4U^y|Db$p5e3MFZZqUU3z=WY`!KbaPLn*~hcq#>J(FK5SoYlh)r"
    "2OC@^G25lXcw8Aqa}Ozf$x{{Ky(D&B~u(3yf|Zx=4HIzr%B=SB)9^D}rLSE2o@}Tw8GNuZXx7oo}ls?Pp+VrG5Vr^rG$V^mxQs^E"
    "q3X?1UbL@wK)6Lmr7~{TojmI4-%NllbHgXj<x2;;0+kcg)SFzZixMH$2*ZIi*n-V#f$^T1zZBz21)7N?{)|`uT-"
    "{D}Qg~AkL;sy5@-"
    "L4A=)`gB0`7axL}Gji@_tDt`BkzD1BO>7n9ceJ@f%0kwzfd4+*3JYUqft74(9b!rqDVbGEkf{eJW@|c<yF5zhHm<F8!iBhL+GzEe"
    "$6UzM=#~%|^t9>*%<SIXn-hbmXYoyd+u4}r-"
    "hLdM!Q9CUuO6}ksr?nF*mY2jxIfsg#?8U|vZ2WY(zFBNmGWIibuZ<yfxt2H5LI&Xsd{#)WHH7y??&iaJX?mMHmXH2hXNx(Oc4$~$"
    "xFq<48Vmk@s9NAlxtxzPaH!bPhz?rl^4HlQZhjMQU=RtDW@<V$<RN&9^!plB1>C~ThIoQ6Xqh(osaIZ@`+su-"
    "{4bUQ9FwT$EdNWF0IV$62*0`#z&x;KqashNIcBqcWNl4?G>)l#ea(P6tz!Rb*KLI70P)B4+U*JT+fZo9wVlOsZ%@P@Eu`=SOUz#+"
    "lcTiUT_djEv(DKV_f!WImL`yRKpge0z6lXNm~)|reSMSRs>@##c)U--"
    "y6w=O?>g&d=i+bmOe<ipdF+YUuc$k$<dSM6<#trFw69e(&i`4+t;trDI-uAy)%@_1=m{gMEn|4MIQ~E&zh=n(jR!{w;#ss{jK!@N"
    "#$FJOXjLw{^AwM~!%kQmHF;#Dbbp|b-$7~<%ia58@D>he)7e(g>2JQ>n<X4_Wki_}uT9067ZAOF;Wsui-"
    "M6{^PF{5(YWr3`IRtA?#z4in0L9|Ojm_&&fl9_Nahi8eJ$9{W+a7N2P3tDowTOI<;^&vxoWIazrwx1>cb(EDXyr8%*Yp|}OZ|Uhe"
    "uoZgasU4SP)h*<6ay3h000O8001EXsBsm?QUCw|SpWb4761SM0000000000q=5hc003`nX=`F{V`y(~FJE72ZfSI1UoLQYP)h*<6"
    "ay3h000O8001EXy15*S!2|#R)C&Lr8UO$Q0000000000q=AM2003`nX=`F{V`y(~FJxhKVP9--VPt7;XD)DgP)h*<6ay3h000O80"
    "01EXz(aXO{R;p9JShMG6#xJL0000000000q=A10003`nX=`F{V`y(~FKuOXa%p38E^v8JO928D0~7!N00;m803iUT_ovQw0{{Rx2"
    "><{V00000000000001_fv*t&0B>w*YhrI>Xm4&WZEs{{Y-"
    "w(1E^v8JO928D0~7!N00;m803iUj51{@h1pol!5dZ)f00000000000001_fmsy*0B>w*YhrI>Xm4&Wb9G{EX>)UFZ*DGdc~DCM0u"
    "%!j000080000X0Iu8j@!tvn08u3X02crN00000000000HlGo8UO%qY-wv^Z)0e0ZZCE;Uu$7!XKZCIaCuNm0Rj{Q6aWAK2mk;8Ap"
    "oJ^o`r%V003x#000^Q0000000000005+c$|V2*Z)|C6VsB$;Z*DJkG+%dVYiV$0WNBe8aCuNm1qJ{B000L72LR>)004wZ00000"
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
