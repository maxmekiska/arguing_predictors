??3
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02unknown8??0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:2(*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:(*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
lstm_3/lstm_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_namelstm_3/lstm_cell_3/kernel
?
-lstm_3/lstm_cell_3/kernel/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_3/kernel*
_output_shapes
:	?*
dtype0
?
#lstm_3/lstm_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?*4
shared_name%#lstm_3/lstm_cell_3/recurrent_kernel
?
7lstm_3/lstm_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_3/lstm_cell_3/recurrent_kernel*
_output_shapes
:	(?*
dtype0
?
lstm_3/lstm_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namelstm_3/lstm_cell_3/bias
?
+lstm_3/lstm_cell_3/bias/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_3/bias*
_output_shapes	
:?*
dtype0
?
lstm_4/lstm_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?**
shared_namelstm_4/lstm_cell_4/kernel
?
-lstm_4/lstm_cell_4/kernel/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_4/kernel*
_output_shapes
:	(?*
dtype0
?
#lstm_4/lstm_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*4
shared_name%#lstm_4/lstm_cell_4/recurrent_kernel
?
7lstm_4/lstm_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_4/lstm_cell_4/recurrent_kernel*
_output_shapes
:	2?*
dtype0
?
lstm_4/lstm_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namelstm_4/lstm_cell_4/bias
?
+lstm_4/lstm_cell_4/bias/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_4/bias*
_output_shapes	
:?*
dtype0
?
lstm_5/lstm_cell_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?**
shared_namelstm_5/lstm_cell_5/kernel
?
-lstm_5/lstm_cell_5/kernel/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_5/kernel*
_output_shapes
:	2?*
dtype0
?
#lstm_5/lstm_cell_5/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*4
shared_name%#lstm_5/lstm_cell_5/recurrent_kernel
?
7lstm_5/lstm_cell_5/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_5/lstm_cell_5/recurrent_kernel*
_output_shapes
:	2?*
dtype0
?
lstm_5/lstm_cell_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namelstm_5/lstm_cell_5/bias
?
+lstm_5/lstm_cell_5/bias/Read/ReadVariableOpReadVariableOplstm_5/lstm_cell_5/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:2(*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:(*
dtype0
?
 Adam/lstm_3/lstm_cell_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/lstm_3/lstm_cell_3/kernel/m
?
4Adam/lstm_3/lstm_cell_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_3/kernel/m*
_output_shapes
:	?*
dtype0
?
*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?*;
shared_name,*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m
?
>Adam/lstm_3/lstm_cell_3/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m*
_output_shapes
:	(?*
dtype0
?
Adam/lstm_3/lstm_cell_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_3/lstm_cell_3/bias/m
?
2Adam/lstm_3/lstm_cell_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_3/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/lstm_4/lstm_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?*1
shared_name" Adam/lstm_4/lstm_cell_4/kernel/m
?
4Adam/lstm_4/lstm_cell_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_4/kernel/m*
_output_shapes
:	(?*
dtype0
?
*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*;
shared_name,*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m
?
>Adam/lstm_4/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m*
_output_shapes
:	2?*
dtype0
?
Adam/lstm_4/lstm_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_4/lstm_cell_4/bias/m
?
2Adam/lstm_4/lstm_cell_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_4/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/lstm_5/lstm_cell_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*1
shared_name" Adam/lstm_5/lstm_cell_5/kernel/m
?
4Adam/lstm_5/lstm_cell_5/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_5/lstm_cell_5/kernel/m*
_output_shapes
:	2?*
dtype0
?
*Adam/lstm_5/lstm_cell_5/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*;
shared_name,*Adam/lstm_5/lstm_cell_5/recurrent_kernel/m
?
>Adam/lstm_5/lstm_cell_5/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_5/lstm_cell_5/recurrent_kernel/m*
_output_shapes
:	2?*
dtype0
?
Adam/lstm_5/lstm_cell_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_5/lstm_cell_5/bias/m
?
2Adam/lstm_5/lstm_cell_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_5/lstm_cell_5/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2(*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:2(*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:(*
dtype0
?
 Adam/lstm_3/lstm_cell_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/lstm_3/lstm_cell_3/kernel/v
?
4Adam/lstm_3/lstm_cell_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_3/kernel/v*
_output_shapes
:	?*
dtype0
?
*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?*;
shared_name,*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v
?
>Adam/lstm_3/lstm_cell_3/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v*
_output_shapes
:	(?*
dtype0
?
Adam/lstm_3/lstm_cell_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_3/lstm_cell_3/bias/v
?
2Adam/lstm_3/lstm_cell_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_3/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/lstm_4/lstm_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(?*1
shared_name" Adam/lstm_4/lstm_cell_4/kernel/v
?
4Adam/lstm_4/lstm_cell_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_4/kernel/v*
_output_shapes
:	(?*
dtype0
?
*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*;
shared_name,*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v
?
>Adam/lstm_4/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v*
_output_shapes
:	2?*
dtype0
?
Adam/lstm_4/lstm_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_4/lstm_cell_4/bias/v
?
2Adam/lstm_4/lstm_cell_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_4/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/lstm_5/lstm_cell_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*1
shared_name" Adam/lstm_5/lstm_cell_5/kernel/v
?
4Adam/lstm_5/lstm_cell_5/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_5/lstm_cell_5/kernel/v*
_output_shapes
:	2?*
dtype0
?
*Adam/lstm_5/lstm_cell_5/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*;
shared_name,*Adam/lstm_5/lstm_cell_5/recurrent_kernel/v
?
>Adam/lstm_5/lstm_cell_5/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_5/lstm_cell_5/recurrent_kernel/v*
_output_shapes
:	2?*
dtype0
?
Adam/lstm_5/lstm_cell_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_5/lstm_cell_5/bias/v
?
2Adam/lstm_5/lstm_cell_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_5/lstm_cell_5/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
?
#iter

$beta_1

%beta_2
	&decay
'learning_ratemsmt(mu)mv*mw+mx,my-mz.m{/m|0m}v~v(v?)v?*v?+v?,v?-v?.v?/v?0v?
 
N
(0
)1
*2
+3
,4
-5
.6
/7
08
9
10
N
(0
)1
*2
+3
,4
-5
.6
/7
08
9
10
?
regularization_losses
	variables

1layers
2metrics
3layer_regularization_losses
trainable_variables
4layer_metrics
5non_trainable_variables
 
~

(kernel
)recurrent_kernel
*bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
 
 

(0
)1
*2

(0
)1
*2
?
regularization_losses
	variables

:layers
;metrics
<layer_regularization_losses
trainable_variables
=layer_metrics

>states
?non_trainable_variables
~

+kernel
,recurrent_kernel
-bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
 
 

+0
,1
-2

+0
,1
-2
?
regularization_losses
	variables

Dlayers
Emetrics
Flayer_regularization_losses
trainable_variables
Glayer_metrics

Hstates
Inon_trainable_variables
~

.kernel
/recurrent_kernel
0bias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
 
 

.0
/1
02

.0
/1
02
?
regularization_losses
	variables

Nlayers
Ometrics
Player_regularization_losses
trainable_variables
Qlayer_metrics

Rstates
Snon_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
 	variables

Tlayers
Ulayer_regularization_losses
Vlayer_metrics
!trainable_variables
Wmetrics
Xnon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_3/lstm_cell_3/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_3/lstm_cell_3/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_3/lstm_cell_3/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_4/lstm_cell_4/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_4/lstm_cell_4/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_4/lstm_cell_4/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_5/lstm_cell_5/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_5/lstm_cell_5/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_5/lstm_cell_5/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

Y0
Z1
 
 
 
 

(0
)1
*2

(0
)1
*2
?
6regularization_losses
7	variables

[layers
\layer_regularization_losses
]layer_metrics
8trainable_variables
^metrics
_non_trainable_variables

0
 
 
 
 
 
 

+0
,1
-2

+0
,1
-2
?
@regularization_losses
A	variables

`layers
alayer_regularization_losses
blayer_metrics
Btrainable_variables
cmetrics
dnon_trainable_variables

0
 
 
 
 
 
 

.0
/1
02

.0
/1
02
?
Jregularization_losses
K	variables

elayers
flayer_regularization_losses
glayer_metrics
Ltrainable_variables
hmetrics
inon_trainable_variables

0
 
 
 
 
 
 
 
 
 
 
4
	jtotal
	kcount
l	variables
m	keras_api
D
	ntotal
	ocount
p
_fn_kwargs
q	variables
r	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

j0
k1

l	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

n0
o1

q	variables
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_3/lstm_cell_3/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_3/lstm_cell_3/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_4/lstm_cell_4/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_4/lstm_cell_4/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_5/lstm_cell_5/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_5/lstm_cell_5/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_5/lstm_cell_5/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_3/lstm_cell_3/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_3/lstm_cell_3/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_4/lstm_cell_4/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_4/lstm_cell_4/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_5/lstm_cell_5/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_5/lstm_cell_5/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_5/lstm_cell_5/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_3_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_3_inputlstm_3/lstm_cell_3/kernel#lstm_3/lstm_cell_3/recurrent_kernellstm_3/lstm_cell_3/biaslstm_4/lstm_cell_4/kernel#lstm_4/lstm_cell_4/recurrent_kernellstm_4/lstm_cell_4/biaslstm_5/lstm_cell_5/kernel#lstm_5/lstm_cell_5/recurrent_kernellstm_5/lstm_cell_5/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_79259
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_3/lstm_cell_3/kernel/Read/ReadVariableOp7lstm_3/lstm_cell_3/recurrent_kernel/Read/ReadVariableOp+lstm_3/lstm_cell_3/bias/Read/ReadVariableOp-lstm_4/lstm_cell_4/kernel/Read/ReadVariableOp7lstm_4/lstm_cell_4/recurrent_kernel/Read/ReadVariableOp+lstm_4/lstm_cell_4/bias/Read/ReadVariableOp-lstm_5/lstm_cell_5/kernel/Read/ReadVariableOp7lstm_5/lstm_cell_5/recurrent_kernel/Read/ReadVariableOp+lstm_5/lstm_cell_5/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_3/kernel/m/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_3/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_3/bias/m/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_4/kernel/m/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_4/bias/m/Read/ReadVariableOp4Adam/lstm_5/lstm_cell_5/kernel/m/Read/ReadVariableOp>Adam/lstm_5/lstm_cell_5/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_5/lstm_cell_5/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_3/kernel/v/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_3/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_3/bias/v/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_4/kernel/v/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_4/bias/v/Read/ReadVariableOp4Adam/lstm_5/lstm_cell_5/kernel/v/Read/ReadVariableOp>Adam/lstm_5/lstm_cell_5/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_5/lstm_cell_5/bias/v/Read/ReadVariableOpConst*7
Tin0
.2,	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_82663
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_3/lstm_cell_3/kernel#lstm_3/lstm_cell_3/recurrent_kernellstm_3/lstm_cell_3/biaslstm_4/lstm_cell_4/kernel#lstm_4/lstm_cell_4/recurrent_kernellstm_4/lstm_cell_4/biaslstm_5/lstm_cell_5/kernel#lstm_5/lstm_cell_5/recurrent_kernellstm_5/lstm_cell_5/biastotalcounttotal_1count_1Adam/dense_1/kernel/mAdam/dense_1/bias/m Adam/lstm_3/lstm_cell_3/kernel/m*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mAdam/lstm_3/lstm_cell_3/bias/m Adam/lstm_4/lstm_cell_4/kernel/m*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mAdam/lstm_4/lstm_cell_4/bias/m Adam/lstm_5/lstm_cell_5/kernel/m*Adam/lstm_5/lstm_cell_5/recurrent_kernel/mAdam/lstm_5/lstm_cell_5/bias/mAdam/dense_1/kernel/vAdam/dense_1/bias/v Adam/lstm_3/lstm_cell_3/kernel/v*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vAdam/lstm_3/lstm_cell_3/bias/v Adam/lstm_4/lstm_cell_4/kernel/v*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vAdam/lstm_4/lstm_cell_4/bias/v Adam/lstm_5/lstm_cell_5/kernel/v*Adam/lstm_5/lstm_cell_5/recurrent_kernel/vAdam/lstm_5/lstm_cell_5/bias/v*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_82799Ȓ/
?	
?
lstm_3_while_cond_79783*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3,
(lstm_3_while_less_lstm_3_strided_slice_1A
=lstm_3_while_lstm_3_while_cond_79783___redundant_placeholder0A
=lstm_3_while_lstm_3_while_cond_79783___redundant_placeholder1A
=lstm_3_while_lstm_3_while_cond_79783___redundant_placeholder2A
=lstm_3_while_lstm_3_while_cond_79783___redundant_placeholder3
lstm_3_while_identity
?
lstm_3/while/LessLesslstm_3_while_placeholder(lstm_3_while_less_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
lstm_3/while/Lessr
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_3/while/Identity"7
lstm_3_while_identitylstm_3/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????(:?????????(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
:
?B
?
lstm_4_while_body_79933*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3)
%lstm_4_while_lstm_4_strided_slice_1_0e
alstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0?
;lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0>
:lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0
lstm_4_while_identity
lstm_4_while_identity_1
lstm_4_while_identity_2
lstm_4_while_identity_3
lstm_4_while_identity_4
lstm_4_while_identity_5'
#lstm_4_while_lstm_4_strided_slice_1c
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor;
7lstm_4_while_lstm_cell_4_matmul_readvariableop_resource=
9lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource<
8lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource??
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2@
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0lstm_4_while_placeholderGlstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????(*
element_dtype022
0lstm_4/while/TensorArrayV2Read/TensorListGetItem?
.lstm_4/while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp9lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	(?*
dtype020
.lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp?
lstm_4/while/lstm_cell_4/MatMulMatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_4/while/lstm_cell_4/MatMul?
0lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp;lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype022
0lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp?
!lstm_4/while/lstm_cell_4/MatMul_1MatMullstm_4_while_placeholder_28lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_4/while/lstm_cell_4/MatMul_1?
lstm_4/while/lstm_cell_4/addAddV2)lstm_4/while/lstm_cell_4/MatMul:product:0+lstm_4/while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_4/while/lstm_cell_4/add?
/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp:lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp?
 lstm_4/while/lstm_cell_4/BiasAddBiasAdd lstm_4/while/lstm_cell_4/add:z:07lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_4/while/lstm_cell_4/BiasAdd?
lstm_4/while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_4/while/lstm_cell_4/Const?
(lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_4/while/lstm_cell_4/split/split_dim?
lstm_4/while/lstm_cell_4/splitSplit1lstm_4/while/lstm_cell_4/split/split_dim:output:0)lstm_4/while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2 
lstm_4/while/lstm_cell_4/split?
 lstm_4/while/lstm_cell_4/SigmoidSigmoid'lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22"
 lstm_4/while/lstm_cell_4/Sigmoid?
"lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid'lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22$
"lstm_4/while/lstm_cell_4/Sigmoid_1?
lstm_4/while/lstm_cell_4/mulMul&lstm_4/while/lstm_cell_4/Sigmoid_1:y:0lstm_4_while_placeholder_3*
T0*'
_output_shapes
:?????????22
lstm_4/while/lstm_cell_4/mul?
lstm_4/while/lstm_cell_4/ReluRelu'lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_4/while/lstm_cell_4/Relu?
lstm_4/while/lstm_cell_4/mul_1Mul$lstm_4/while/lstm_cell_4/Sigmoid:y:0+lstm_4/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22 
lstm_4/while/lstm_cell_4/mul_1?
lstm_4/while/lstm_cell_4/add_1AddV2 lstm_4/while/lstm_cell_4/mul:z:0"lstm_4/while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22 
lstm_4/while/lstm_cell_4/add_1?
"lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid'lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22$
"lstm_4/while/lstm_cell_4/Sigmoid_2?
lstm_4/while/lstm_cell_4/Relu_1Relu"lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22!
lstm_4/while/lstm_cell_4/Relu_1?
lstm_4/while/lstm_cell_4/mul_2Mul&lstm_4/while/lstm_cell_4/Sigmoid_2:y:0-lstm_4/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22 
lstm_4/while/lstm_cell_4/mul_2?
1lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_4_while_placeholder_1lstm_4_while_placeholder"lstm_4/while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_4/while/TensorArrayV2Write/TensorListSetItemj
lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_4/while/add/y?
lstm_4/while/addAddV2lstm_4_while_placeholderlstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_4/while/addn
lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_4/while/add_1/y?
lstm_4/while/add_1AddV2&lstm_4_while_lstm_4_while_loop_counterlstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_4/while/add_1s
lstm_4/while/IdentityIdentitylstm_4/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_4/while/Identity?
lstm_4/while/Identity_1Identity,lstm_4_while_lstm_4_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_4/while/Identity_1u
lstm_4/while/Identity_2Identitylstm_4/while/add:z:0*
T0*
_output_shapes
: 2
lstm_4/while/Identity_2?
lstm_4/while/Identity_3IdentityAlstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_4/while/Identity_3?
lstm_4/while/Identity_4Identity"lstm_4/while/lstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????22
lstm_4/while/Identity_4?
lstm_4/while/Identity_5Identity"lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_4/while/Identity_5"7
lstm_4_while_identitylstm_4/while/Identity:output:0";
lstm_4_while_identity_1 lstm_4/while/Identity_1:output:0";
lstm_4_while_identity_2 lstm_4/while/Identity_2:output:0";
lstm_4_while_identity_3 lstm_4/while/Identity_3:output:0";
lstm_4_while_identity_4 lstm_4/while/Identity_4:output:0";
lstm_4_while_identity_5 lstm_4/while/Identity_5:output:0"L
#lstm_4_while_lstm_4_strided_slice_1%lstm_4_while_lstm_4_strided_slice_1_0"v
8lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource:lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0"x
9lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource;lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0"t
7lstm_4_while_lstm_cell_4_matmul_readvariableop_resource9lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0"?
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensoralstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_lstm_3_layer_call_fn_80883

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????(*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_783502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_lstm_5_layer_call_fn_81856
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_779002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????2
"
_user_specified_name
inputs/0
?
?
+__inference_lstm_cell_4_layer_call_fn_82414

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_769272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????(:?????????2:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?	
?
,__inference_sequential_1_layer_call_fn_79165
lstm_3_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_791402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namelstm_3_input
?$
?
while_body_77221
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_4_77245_0
while_lstm_cell_4_77247_0
while_lstm_cell_4_77249_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_4_77245
while_lstm_cell_4_77247
while_lstm_cell_4_77249??)while/lstm_cell_4/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????(*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_77245_0while_lstm_cell_4_77247_0while_lstm_cell_4_77249_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_768942+
)while/lstm_cell_4/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1*^while/lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2*^while/lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_4_77245while_lstm_cell_4_77245_0"4
while_lstm_cell_4_77247while_lstm_cell_4_77247_0"4
while_lstm_cell_4_77249while_lstm_cell_4_77249_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_76927

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????(:?????????2:?????????2::::O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_79060

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2:::O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
while_cond_77352
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_77352___redundant_placeholder03
/while_while_cond_77352___redundant_placeholder13
/while_while_cond_77352___redundant_placeholder23
/while_while_cond_77352___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?W
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_80533
inputs_0.
*lstm_cell_3_matmul_readvariableop_resource0
,lstm_cell_3_matmul_1_readvariableop_resource/
+lstm_cell_3_biasadd_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????(2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????(2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!lstm_cell_3/MatMul/ReadVariableOp?
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul?
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	(?*
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp?
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_1?
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add?
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAddh
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/Const|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dim?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
lstm_cell_3/split?
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid?
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid_1?
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mulz
lstm_cell_3/ReluRelulstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Relu?
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mul_1?
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/add_1?
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid_2y
lstm_cell_3/Relu_1Relulstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Relu_1?
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0 lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????(:?????????(: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_80448*
condR
while_cond_80447*K
output_shapes:
8: : : : :?????????(:?????????(: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????(*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????(2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :??????????????????(2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?W
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_81845
inputs_0.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_81760*
condR
while_cond_81759*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????2
"
_user_specified_name
inputs/0
?D
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_78032

inputs
lstm_cell_5_77950
lstm_cell_5_77952
lstm_cell_5_77954
identity??#lstm_cell_5/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5_77950lstm_cell_5_77952lstm_cell_5_77954*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_775372%
#lstm_cell_5/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5_77950lstm_cell_5_77952lstm_cell_5_77954*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_77963*
condR
while_cond_77962*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_5/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::2J
#lstm_cell_5/StatefulPartitionedCall#lstm_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????2
 
_user_specified_nameinputs
?W
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_79020

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_78935*
condR
while_cond_78934*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????2:::2
whilewhile:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
while_cond_80775
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_80775___redundant_placeholder03
/while_while_cond_80775___redundant_placeholder13
/while_while_cond_80775___redundant_placeholder23
/while_while_cond_80775___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????(:?????????(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
:
?9
?
while_body_81104
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_4_matmul_readvariableop_resource_08
4while_lstm_cell_4_matmul_1_readvariableop_resource_07
3while_lstm_cell_4_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_4_matmul_readvariableop_resource6
2while_lstm_cell_4_matmul_1_readvariableop_resource5
1while_lstm_cell_4_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????(*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	(?*
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/MatMul?
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp?
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/MatMul_1?
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/add?
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp?
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/BiasAddt
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const?
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_4/split?
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid?
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid_1?
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul?
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Relu?
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul_1?
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/add_1?
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid_2?
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Relu_1?
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?W
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_81692
inputs_0.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_81607*
condR
while_cond_81606*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????2
"
_user_specified_name
inputs/0
?
?
#__inference_signature_wrapper_79259
lstm_3_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_762112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namelstm_3_input
?
?
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_82380

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????(:?????????2:?????????2::::O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?W
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_81036
inputs_0.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????(2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/MatMul?
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp?
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/MatMul_1?
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/add?
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dim?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_4/split?
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid?
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid_1?
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Relu?
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mul_1?
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/add_1?
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Relu_1?
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_80951*
condR
while_cond_80950*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????(:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????(
"
_user_specified_name
inputs/0
?9
?
while_body_80623
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_3_matmul_readvariableop_resource_08
4while_lstm_cell_3_matmul_1_readvariableop_resource_07
3while_lstm_cell_3_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_3_matmul_readvariableop_resource6
2while_lstm_cell_3_matmul_1_readvariableop_resource5
1while_lstm_cell_3_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOp?
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul?
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	(?*
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOp?
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_1?
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add?
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOp?
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAddt
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_3/Const?
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
while/lstm_cell_3/split?
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid?
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid_1?
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul?
while/lstm_cell_3/ReluRelu while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Relu?
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0$while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul_1?
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/add_1?
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid_2?
while/lstm_cell_3/Relu_1Reluwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Relu_1?
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0&while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0*
T0*'
_output_shapes
:?????????(2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????(:?????????(: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_78599
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_78599___redundant_placeholder03
/while_while_cond_78599___redundant_placeholder13
/while_while_cond_78599___redundant_placeholder23
/while_while_cond_78599___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?9
?
while_body_78112
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_3_matmul_readvariableop_resource_08
4while_lstm_cell_3_matmul_1_readvariableop_resource_07
3while_lstm_cell_3_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_3_matmul_readvariableop_resource6
2while_lstm_cell_3_matmul_1_readvariableop_resource5
1while_lstm_cell_3_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOp?
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul?
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	(?*
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOp?
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_1?
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add?
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOp?
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAddt
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_3/Const?
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
while/lstm_cell_3/split?
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid?
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid_1?
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul?
while/lstm_cell_3/ReluRelu while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Relu?
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0$while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul_1?
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/add_1?
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid_2?
while/lstm_cell_3/Relu_1Reluwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Relu_1?
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0&while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0*
T0*'
_output_shapes
:?????????(2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????(:?????????(: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_80622
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_80622___redundant_placeholder03
/while_while_cond_80622___redundant_placeholder13
/while_while_cond_80622___redundant_placeholder23
/while_while_cond_80622___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????(:?????????(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
:
?
?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_82280

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	(?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????(2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????(2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????(2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????(2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????(2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????(2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????(2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????(2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????(2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????(2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????(2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????(:?????????(::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????(
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????(
"
_user_specified_name
states/1
?B
?
lstm_5_while_body_79625*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0?
;lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0>
:lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0
lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor;
7lstm_5_while_lstm_cell_5_matmul_readvariableop_resource=
9lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource<
8lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource??
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2@
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype022
0lstm_5/while/TensorArrayV2Read/TensorListGetItem?
.lstm_5/while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp9lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	2?*
dtype020
.lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp?
lstm_5/while/lstm_cell_5/MatMulMatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_5/while/lstm_cell_5/MatMul?
0lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp;lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype022
0lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp?
!lstm_5/while/lstm_cell_5/MatMul_1MatMullstm_5_while_placeholder_28lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_5/while/lstm_cell_5/MatMul_1?
lstm_5/while/lstm_cell_5/addAddV2)lstm_5/while/lstm_cell_5/MatMul:product:0+lstm_5/while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_5/while/lstm_cell_5/add?
/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp:lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp?
 lstm_5/while/lstm_cell_5/BiasAddBiasAdd lstm_5/while/lstm_cell_5/add:z:07lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_5/while/lstm_cell_5/BiasAdd?
lstm_5/while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_5/while/lstm_cell_5/Const?
(lstm_5/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_5/while/lstm_cell_5/split/split_dim?
lstm_5/while/lstm_cell_5/splitSplit1lstm_5/while/lstm_cell_5/split/split_dim:output:0)lstm_5/while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2 
lstm_5/while/lstm_cell_5/split?
 lstm_5/while/lstm_cell_5/SigmoidSigmoid'lstm_5/while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22"
 lstm_5/while/lstm_cell_5/Sigmoid?
"lstm_5/while/lstm_cell_5/Sigmoid_1Sigmoid'lstm_5/while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22$
"lstm_5/while/lstm_cell_5/Sigmoid_1?
lstm_5/while/lstm_cell_5/mulMul&lstm_5/while/lstm_cell_5/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*'
_output_shapes
:?????????22
lstm_5/while/lstm_cell_5/mul?
lstm_5/while/lstm_cell_5/ReluRelu'lstm_5/while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_5/while/lstm_cell_5/Relu?
lstm_5/while/lstm_cell_5/mul_1Mul$lstm_5/while/lstm_cell_5/Sigmoid:y:0+lstm_5/while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22 
lstm_5/while/lstm_cell_5/mul_1?
lstm_5/while/lstm_cell_5/add_1AddV2 lstm_5/while/lstm_cell_5/mul:z:0"lstm_5/while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22 
lstm_5/while/lstm_cell_5/add_1?
"lstm_5/while/lstm_cell_5/Sigmoid_2Sigmoid'lstm_5/while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22$
"lstm_5/while/lstm_cell_5/Sigmoid_2?
lstm_5/while/lstm_cell_5/Relu_1Relu"lstm_5/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22!
lstm_5/while/lstm_cell_5/Relu_1?
lstm_5/while/lstm_cell_5/mul_2Mul&lstm_5/while/lstm_cell_5/Sigmoid_2:y:0-lstm_5/while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22 
lstm_5/while/lstm_cell_5/mul_2?
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1lstm_5_while_placeholder"lstm_5/while/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_5/while/TensorArrayV2Write/TensorListSetItemj
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add/y?
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/addn
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add_1/y?
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/add_1s
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_5/while/Identity?
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_5/while/Identity_1u
lstm_5/while/Identity_2Identitylstm_5/while/add:z:0*
T0*
_output_shapes
: 2
lstm_5/while/Identity_2?
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_5/while/Identity_3?
lstm_5/while/Identity_4Identity"lstm_5/while/lstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/while/Identity_4?
lstm_5/while/Identity_5Identity"lstm_5/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/while/Identity_5"7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"v
8lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource:lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0"x
9lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource;lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0"t
7lstm_5_while_lstm_cell_5_matmul_readvariableop_resource9lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0"?
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?9
?
while_body_78935
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?W
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_81189
inputs_0.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????(2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/MatMul?
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp?
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/MatMul_1?
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/add?
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dim?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_4/split?
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid?
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid_1?
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Relu?
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mul_1?
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/add_1?
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Relu_1?
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_81104*
condR
while_cond_81103*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????(:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????(
"
_user_specified_name
inputs/0
?9
?
while_body_81432
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_4_matmul_readvariableop_resource_08
4while_lstm_cell_4_matmul_1_readvariableop_resource_07
3while_lstm_cell_4_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_4_matmul_readvariableop_resource6
2while_lstm_cell_4_matmul_1_readvariableop_resource5
1while_lstm_cell_4_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????(*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	(?*
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/MatMul?
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp?
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/MatMul_1?
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/add?
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp?
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/BiasAddt
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const?
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_4/split?
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid?
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid_1?
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul?
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Relu?
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul_1?
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/add_1?
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid_2?
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Relu_1?
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_77537

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????2:?????????2:?????????2::::O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
?9
?
while_body_78265
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_3_matmul_readvariableop_resource_08
4while_lstm_cell_3_matmul_1_readvariableop_resource_07
3while_lstm_cell_3_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_3_matmul_readvariableop_resource6
2while_lstm_cell_3_matmul_1_readvariableop_resource5
1while_lstm_cell_3_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOp?
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul?
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	(?*
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOp?
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_1?
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add?
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOp?
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAddt
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_3/Const?
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
while/lstm_cell_3/split?
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid?
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid_1?
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul?
while/lstm_cell_3/ReluRelu while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Relu?
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0$while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul_1?
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/add_1?
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid_2?
while/lstm_cell_3/Relu_1Reluwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Relu_1?
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0&while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0*
T0*'
_output_shapes
:?????????(2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????(:?????????(: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_76284

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	(?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????(2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????(2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????(2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????(2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????(2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????(2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????(2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????(2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????(2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????(2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????(2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????(:?????????(::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????(
 
_user_specified_namestates:OK
'
_output_shapes
:?????????(
 
_user_specified_namestates
?
?
while_cond_77962
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_77962___redundant_placeholder03
/while_while_cond_77962___redundant_placeholder13
/while_while_cond_77962___redundant_placeholder23
/while_while_cond_77962___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
+__inference_lstm_cell_3_layer_call_fn_82314

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????(:?????????(:?????????(*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_763172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????(:?????????(:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????(
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????(
"
_user_specified_name
states/1
?
?
while_cond_80950
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_80950___redundant_placeholder03
/while_while_cond_80950___redundant_placeholder13
/while_while_cond_80950___redundant_placeholder23
/while_while_cond_80950___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
+__inference_lstm_cell_3_layer_call_fn_82297

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????(:?????????(:?????????(*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_762842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????(:?????????(:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????(
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????(
"
_user_specified_name
states/1
?R
?
$sequential_1_lstm_4_while_body_75971D
@sequential_1_lstm_4_while_sequential_1_lstm_4_while_loop_counterJ
Fsequential_1_lstm_4_while_sequential_1_lstm_4_while_maximum_iterations)
%sequential_1_lstm_4_while_placeholder+
'sequential_1_lstm_4_while_placeholder_1+
'sequential_1_lstm_4_while_placeholder_2+
'sequential_1_lstm_4_while_placeholder_3C
?sequential_1_lstm_4_while_sequential_1_lstm_4_strided_slice_1_0
{sequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensor_0J
Fsequential_1_lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0L
Hsequential_1_lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0K
Gsequential_1_lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0&
"sequential_1_lstm_4_while_identity(
$sequential_1_lstm_4_while_identity_1(
$sequential_1_lstm_4_while_identity_2(
$sequential_1_lstm_4_while_identity_3(
$sequential_1_lstm_4_while_identity_4(
$sequential_1_lstm_4_while_identity_5A
=sequential_1_lstm_4_while_sequential_1_lstm_4_strided_slice_1}
ysequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensorH
Dsequential_1_lstm_4_while_lstm_cell_4_matmul_readvariableop_resourceJ
Fsequential_1_lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resourceI
Esequential_1_lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource??
Ksequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2M
Ksequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=sequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_4_while_placeholderTsequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????(*
element_dtype02?
=sequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem?
;sequential_1/lstm_4/while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpFsequential_1_lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	(?*
dtype02=
;sequential_1/lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp?
,sequential_1/lstm_4/while/lstm_cell_4/MatMulMatMulDsequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_1/lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,sequential_1/lstm_4/while/lstm_cell_4/MatMul?
=sequential_1/lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpHsequential_1_lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02?
=sequential_1/lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp?
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_1MatMul'sequential_1_lstm_4_while_placeholder_2Esequential_1/lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_1?
)sequential_1/lstm_4/while/lstm_cell_4/addAddV26sequential_1/lstm_4/while/lstm_cell_4/MatMul:product:08sequential_1/lstm_4/while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2+
)sequential_1/lstm_4/while/lstm_cell_4/add?
<sequential_1/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOpGsequential_1_lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02>
<sequential_1/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp?
-sequential_1/lstm_4/while/lstm_cell_4/BiasAddBiasAdd-sequential_1/lstm_4/while/lstm_cell_4/add:z:0Dsequential_1/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-sequential_1/lstm_4/while/lstm_cell_4/BiasAdd?
+sequential_1/lstm_4/while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+sequential_1/lstm_4/while/lstm_cell_4/Const?
5sequential_1/lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_1/lstm_4/while/lstm_cell_4/split/split_dim?
+sequential_1/lstm_4/while/lstm_cell_4/splitSplit>sequential_1/lstm_4/while/lstm_cell_4/split/split_dim:output:06sequential_1/lstm_4/while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2-
+sequential_1/lstm_4/while/lstm_cell_4/split?
-sequential_1/lstm_4/while/lstm_cell_4/SigmoidSigmoid4sequential_1/lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22/
-sequential_1/lstm_4/while/lstm_cell_4/Sigmoid?
/sequential_1/lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid4sequential_1/lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????221
/sequential_1/lstm_4/while/lstm_cell_4/Sigmoid_1?
)sequential_1/lstm_4/while/lstm_cell_4/mulMul3sequential_1/lstm_4/while/lstm_cell_4/Sigmoid_1:y:0'sequential_1_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:?????????22+
)sequential_1/lstm_4/while/lstm_cell_4/mul?
*sequential_1/lstm_4/while/lstm_cell_4/ReluRelu4sequential_1/lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22,
*sequential_1/lstm_4/while/lstm_cell_4/Relu?
+sequential_1/lstm_4/while/lstm_cell_4/mul_1Mul1sequential_1/lstm_4/while/lstm_cell_4/Sigmoid:y:08sequential_1/lstm_4/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22-
+sequential_1/lstm_4/while/lstm_cell_4/mul_1?
+sequential_1/lstm_4/while/lstm_cell_4/add_1AddV2-sequential_1/lstm_4/while/lstm_cell_4/mul:z:0/sequential_1/lstm_4/while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22-
+sequential_1/lstm_4/while/lstm_cell_4/add_1?
/sequential_1/lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid4sequential_1/lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????221
/sequential_1/lstm_4/while/lstm_cell_4/Sigmoid_2?
,sequential_1/lstm_4/while/lstm_cell_4/Relu_1Relu/sequential_1/lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22.
,sequential_1/lstm_4/while/lstm_cell_4/Relu_1?
+sequential_1/lstm_4/while/lstm_cell_4/mul_2Mul3sequential_1/lstm_4/while/lstm_cell_4/Sigmoid_2:y:0:sequential_1/lstm_4/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22-
+sequential_1/lstm_4/while/lstm_cell_4/mul_2?
>sequential_1/lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_4_while_placeholder_1%sequential_1_lstm_4_while_placeholder/sequential_1/lstm_4/while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_1/lstm_4/while/TensorArrayV2Write/TensorListSetItem?
sequential_1/lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_1/lstm_4/while/add/y?
sequential_1/lstm_4/while/addAddV2%sequential_1_lstm_4_while_placeholder(sequential_1/lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_4/while/add?
!sequential_1/lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_1/lstm_4/while/add_1/y?
sequential_1/lstm_4/while/add_1AddV2@sequential_1_lstm_4_while_sequential_1_lstm_4_while_loop_counter*sequential_1/lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_4/while/add_1?
"sequential_1/lstm_4/while/IdentityIdentity#sequential_1/lstm_4/while/add_1:z:0*
T0*
_output_shapes
: 2$
"sequential_1/lstm_4/while/Identity?
$sequential_1/lstm_4/while/Identity_1IdentityFsequential_1_lstm_4_while_sequential_1_lstm_4_while_maximum_iterations*
T0*
_output_shapes
: 2&
$sequential_1/lstm_4/while/Identity_1?
$sequential_1/lstm_4/while/Identity_2Identity!sequential_1/lstm_4/while/add:z:0*
T0*
_output_shapes
: 2&
$sequential_1/lstm_4/while/Identity_2?
$sequential_1/lstm_4/while/Identity_3IdentityNsequential_1/lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2&
$sequential_1/lstm_4/while/Identity_3?
$sequential_1/lstm_4/while/Identity_4Identity/sequential_1/lstm_4/while/lstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????22&
$sequential_1/lstm_4/while/Identity_4?
$sequential_1/lstm_4/while/Identity_5Identity/sequential_1/lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22&
$sequential_1/lstm_4/while/Identity_5"Q
"sequential_1_lstm_4_while_identity+sequential_1/lstm_4/while/Identity:output:0"U
$sequential_1_lstm_4_while_identity_1-sequential_1/lstm_4/while/Identity_1:output:0"U
$sequential_1_lstm_4_while_identity_2-sequential_1/lstm_4/while/Identity_2:output:0"U
$sequential_1_lstm_4_while_identity_3-sequential_1/lstm_4/while/Identity_3:output:0"U
$sequential_1_lstm_4_while_identity_4-sequential_1/lstm_4/while/Identity_4:output:0"U
$sequential_1_lstm_4_while_identity_5-sequential_1/lstm_4/while/Identity_5:output:0"?
Esequential_1_lstm_4_while_lstm_cell_4_biasadd_readvariableop_resourceGsequential_1_lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0"?
Fsequential_1_lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resourceHsequential_1_lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0"?
Dsequential_1_lstm_4_while_lstm_cell_4_matmul_readvariableop_resourceFsequential_1_lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0"?
=sequential_1_lstm_4_while_sequential_1_lstm_4_strided_slice_1?sequential_1_lstm_4_while_sequential_1_lstm_4_strided_slice_1_0"?
ysequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_81606
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_81606___redundant_placeholder03
/while_while_cond_81606___redundant_placeholder13
/while_while_cond_81606___redundant_placeholder23
/while_while_cond_81606___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?B
?
lstm_5_while_body_80082*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3)
%lstm_5_while_lstm_5_strided_slice_1_0e
alstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0?
;lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0>
:lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0
lstm_5_while_identity
lstm_5_while_identity_1
lstm_5_while_identity_2
lstm_5_while_identity_3
lstm_5_while_identity_4
lstm_5_while_identity_5'
#lstm_5_while_lstm_5_strided_slice_1c
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor;
7lstm_5_while_lstm_cell_5_matmul_readvariableop_resource=
9lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource<
8lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource??
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2@
>lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0lstm_5_while_placeholderGlstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype022
0lstm_5/while/TensorArrayV2Read/TensorListGetItem?
.lstm_5/while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp9lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	2?*
dtype020
.lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp?
lstm_5/while/lstm_cell_5/MatMulMatMul7lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_5/while/lstm_cell_5/MatMul?
0lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp;lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype022
0lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp?
!lstm_5/while/lstm_cell_5/MatMul_1MatMullstm_5_while_placeholder_28lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_5/while/lstm_cell_5/MatMul_1?
lstm_5/while/lstm_cell_5/addAddV2)lstm_5/while/lstm_cell_5/MatMul:product:0+lstm_5/while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_5/while/lstm_cell_5/add?
/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp:lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp?
 lstm_5/while/lstm_cell_5/BiasAddBiasAdd lstm_5/while/lstm_cell_5/add:z:07lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_5/while/lstm_cell_5/BiasAdd?
lstm_5/while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_5/while/lstm_cell_5/Const?
(lstm_5/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_5/while/lstm_cell_5/split/split_dim?
lstm_5/while/lstm_cell_5/splitSplit1lstm_5/while/lstm_cell_5/split/split_dim:output:0)lstm_5/while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2 
lstm_5/while/lstm_cell_5/split?
 lstm_5/while/lstm_cell_5/SigmoidSigmoid'lstm_5/while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22"
 lstm_5/while/lstm_cell_5/Sigmoid?
"lstm_5/while/lstm_cell_5/Sigmoid_1Sigmoid'lstm_5/while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22$
"lstm_5/while/lstm_cell_5/Sigmoid_1?
lstm_5/while/lstm_cell_5/mulMul&lstm_5/while/lstm_cell_5/Sigmoid_1:y:0lstm_5_while_placeholder_3*
T0*'
_output_shapes
:?????????22
lstm_5/while/lstm_cell_5/mul?
lstm_5/while/lstm_cell_5/ReluRelu'lstm_5/while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_5/while/lstm_cell_5/Relu?
lstm_5/while/lstm_cell_5/mul_1Mul$lstm_5/while/lstm_cell_5/Sigmoid:y:0+lstm_5/while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22 
lstm_5/while/lstm_cell_5/mul_1?
lstm_5/while/lstm_cell_5/add_1AddV2 lstm_5/while/lstm_cell_5/mul:z:0"lstm_5/while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22 
lstm_5/while/lstm_cell_5/add_1?
"lstm_5/while/lstm_cell_5/Sigmoid_2Sigmoid'lstm_5/while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22$
"lstm_5/while/lstm_cell_5/Sigmoid_2?
lstm_5/while/lstm_cell_5/Relu_1Relu"lstm_5/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22!
lstm_5/while/lstm_cell_5/Relu_1?
lstm_5/while/lstm_cell_5/mul_2Mul&lstm_5/while/lstm_cell_5/Sigmoid_2:y:0-lstm_5/while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22 
lstm_5/while/lstm_cell_5/mul_2?
1lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_5_while_placeholder_1lstm_5_while_placeholder"lstm_5/while/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_5/while/TensorArrayV2Write/TensorListSetItemj
lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add/y?
lstm_5/while/addAddV2lstm_5_while_placeholderlstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/addn
lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/while/add_1/y?
lstm_5/while/add_1AddV2&lstm_5_while_lstm_5_while_loop_counterlstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_5/while/add_1s
lstm_5/while/IdentityIdentitylstm_5/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_5/while/Identity?
lstm_5/while/Identity_1Identity,lstm_5_while_lstm_5_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_5/while/Identity_1u
lstm_5/while/Identity_2Identitylstm_5/while/add:z:0*
T0*
_output_shapes
: 2
lstm_5/while/Identity_2?
lstm_5/while/Identity_3IdentityAlstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_5/while/Identity_3?
lstm_5/while/Identity_4Identity"lstm_5/while/lstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/while/Identity_4?
lstm_5/while/Identity_5Identity"lstm_5/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/while/Identity_5"7
lstm_5_while_identitylstm_5/while/Identity:output:0";
lstm_5_while_identity_1 lstm_5/while/Identity_1:output:0";
lstm_5_while_identity_2 lstm_5/while/Identity_2:output:0";
lstm_5_while_identity_3 lstm_5/while/Identity_3:output:0";
lstm_5_while_identity_4 lstm_5/while/Identity_4:output:0";
lstm_5_while_identity_5 lstm_5/while/Identity_5:output:0"L
#lstm_5_while_lstm_5_strided_slice_1%lstm_5_while_lstm_5_strided_slice_1_0"v
8lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource:lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0"x
9lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource;lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0"t
7lstm_5_while_lstm_cell_5_matmul_readvariableop_resource9lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0"?
_lstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensoralstm_5_while_tensorarrayv2read_tensorlistgetitem_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
??
?
 __inference__wrapped_model_76211
lstm_3_inputB
>sequential_1_lstm_3_lstm_cell_3_matmul_readvariableop_resourceD
@sequential_1_lstm_3_lstm_cell_3_matmul_1_readvariableop_resourceC
?sequential_1_lstm_3_lstm_cell_3_biasadd_readvariableop_resourceB
>sequential_1_lstm_4_lstm_cell_4_matmul_readvariableop_resourceD
@sequential_1_lstm_4_lstm_cell_4_matmul_1_readvariableop_resourceC
?sequential_1_lstm_4_lstm_cell_4_biasadd_readvariableop_resourceB
>sequential_1_lstm_5_lstm_cell_5_matmul_readvariableop_resourceD
@sequential_1_lstm_5_lstm_cell_5_matmul_1_readvariableop_resourceC
?sequential_1_lstm_5_lstm_cell_5_biasadd_readvariableop_resource7
3sequential_1_dense_1_matmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identity??sequential_1/lstm_3/while?sequential_1/lstm_4/while?sequential_1/lstm_5/whiler
sequential_1/lstm_3/ShapeShapelstm_3_input*
T0*
_output_shapes
:2
sequential_1/lstm_3/Shape?
'sequential_1/lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_1/lstm_3/strided_slice/stack?
)sequential_1/lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_3/strided_slice/stack_1?
)sequential_1/lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_3/strided_slice/stack_2?
!sequential_1/lstm_3/strided_sliceStridedSlice"sequential_1/lstm_3/Shape:output:00sequential_1/lstm_3/strided_slice/stack:output:02sequential_1/lstm_3/strided_slice/stack_1:output:02sequential_1/lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_1/lstm_3/strided_slice?
sequential_1/lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2!
sequential_1/lstm_3/zeros/mul/y?
sequential_1/lstm_3/zeros/mulMul*sequential_1/lstm_3/strided_slice:output:0(sequential_1/lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_3/zeros/mul?
 sequential_1/lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_1/lstm_3/zeros/Less/y?
sequential_1/lstm_3/zeros/LessLess!sequential_1/lstm_3/zeros/mul:z:0)sequential_1/lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_1/lstm_3/zeros/Less?
"sequential_1/lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2$
"sequential_1/lstm_3/zeros/packed/1?
 sequential_1/lstm_3/zeros/packedPack*sequential_1/lstm_3/strided_slice:output:0+sequential_1/lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_1/lstm_3/zeros/packed?
sequential_1/lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_1/lstm_3/zeros/Const?
sequential_1/lstm_3/zerosFill)sequential_1/lstm_3/zeros/packed:output:0(sequential_1/lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:?????????(2
sequential_1/lstm_3/zeros?
!sequential_1/lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2#
!sequential_1/lstm_3/zeros_1/mul/y?
sequential_1/lstm_3/zeros_1/mulMul*sequential_1/lstm_3/strided_slice:output:0*sequential_1/lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_3/zeros_1/mul?
"sequential_1/lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_1/lstm_3/zeros_1/Less/y?
 sequential_1/lstm_3/zeros_1/LessLess#sequential_1/lstm_3/zeros_1/mul:z:0+sequential_1/lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_1/lstm_3/zeros_1/Less?
$sequential_1/lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2&
$sequential_1/lstm_3/zeros_1/packed/1?
"sequential_1/lstm_3/zeros_1/packedPack*sequential_1/lstm_3/strided_slice:output:0-sequential_1/lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/lstm_3/zeros_1/packed?
!sequential_1/lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_1/lstm_3/zeros_1/Const?
sequential_1/lstm_3/zeros_1Fill+sequential_1/lstm_3/zeros_1/packed:output:0*sequential_1/lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????(2
sequential_1/lstm_3/zeros_1?
"sequential_1/lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_1/lstm_3/transpose/perm?
sequential_1/lstm_3/transpose	Transposelstm_3_input+sequential_1/lstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
sequential_1/lstm_3/transpose?
sequential_1/lstm_3/Shape_1Shape!sequential_1/lstm_3/transpose:y:0*
T0*
_output_shapes
:2
sequential_1/lstm_3/Shape_1?
)sequential_1/lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_3/strided_slice_1/stack?
+sequential_1/lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_1/stack_1?
+sequential_1/lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_1/stack_2?
#sequential_1/lstm_3/strided_slice_1StridedSlice$sequential_1/lstm_3/Shape_1:output:02sequential_1/lstm_3/strided_slice_1/stack:output:04sequential_1/lstm_3/strided_slice_1/stack_1:output:04sequential_1/lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_1/lstm_3/strided_slice_1?
/sequential_1/lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_1/lstm_3/TensorArrayV2/element_shape?
!sequential_1/lstm_3/TensorArrayV2TensorListReserve8sequential_1/lstm_3/TensorArrayV2/element_shape:output:0,sequential_1/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_1/lstm_3/TensorArrayV2?
Isequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2K
Isequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape?
;sequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_3/transpose:y:0Rsequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor?
)sequential_1/lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_3/strided_slice_2/stack?
+sequential_1/lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_2/stack_1?
+sequential_1/lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_2/stack_2?
#sequential_1/lstm_3/strided_slice_2StridedSlice!sequential_1/lstm_3/transpose:y:02sequential_1/lstm_3/strided_slice_2/stack:output:04sequential_1/lstm_3/strided_slice_2/stack_1:output:04sequential_1/lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2%
#sequential_1/lstm_3/strided_slice_2?
5sequential_1/lstm_3/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp>sequential_1_lstm_3_lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype027
5sequential_1/lstm_3/lstm_cell_3/MatMul/ReadVariableOp?
&sequential_1/lstm_3/lstm_cell_3/MatMulMatMul,sequential_1/lstm_3/strided_slice_2:output:0=sequential_1/lstm_3/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential_1/lstm_3/lstm_cell_3/MatMul?
7sequential_1/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp@sequential_1_lstm_3_lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	(?*
dtype029
7sequential_1/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp?
(sequential_1/lstm_3/lstm_cell_3/MatMul_1MatMul"sequential_1/lstm_3/zeros:output:0?sequential_1/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_1/lstm_3/lstm_cell_3/MatMul_1?
#sequential_1/lstm_3/lstm_cell_3/addAddV20sequential_1/lstm_3/lstm_cell_3/MatMul:product:02sequential_1/lstm_3/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2%
#sequential_1/lstm_3/lstm_cell_3/add?
6sequential_1/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential_1/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp?
'sequential_1/lstm_3/lstm_cell_3/BiasAddBiasAdd'sequential_1/lstm_3/lstm_cell_3/add:z:0>sequential_1/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_1/lstm_3/lstm_cell_3/BiasAdd?
%sequential_1/lstm_3/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/lstm_3/lstm_cell_3/Const?
/sequential_1/lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_1/lstm_3/lstm_cell_3/split/split_dim?
%sequential_1/lstm_3/lstm_cell_3/splitSplit8sequential_1/lstm_3/lstm_cell_3/split/split_dim:output:00sequential_1/lstm_3/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2'
%sequential_1/lstm_3/lstm_cell_3/split?
'sequential_1/lstm_3/lstm_cell_3/SigmoidSigmoid.sequential_1/lstm_3/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2)
'sequential_1/lstm_3/lstm_cell_3/Sigmoid?
)sequential_1/lstm_3/lstm_cell_3/Sigmoid_1Sigmoid.sequential_1/lstm_3/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2+
)sequential_1/lstm_3/lstm_cell_3/Sigmoid_1?
#sequential_1/lstm_3/lstm_cell_3/mulMul-sequential_1/lstm_3/lstm_cell_3/Sigmoid_1:y:0$sequential_1/lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:?????????(2%
#sequential_1/lstm_3/lstm_cell_3/mul?
$sequential_1/lstm_3/lstm_cell_3/ReluRelu.sequential_1/lstm_3/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2&
$sequential_1/lstm_3/lstm_cell_3/Relu?
%sequential_1/lstm_3/lstm_cell_3/mul_1Mul+sequential_1/lstm_3/lstm_cell_3/Sigmoid:y:02sequential_1/lstm_3/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2'
%sequential_1/lstm_3/lstm_cell_3/mul_1?
%sequential_1/lstm_3/lstm_cell_3/add_1AddV2'sequential_1/lstm_3/lstm_cell_3/mul:z:0)sequential_1/lstm_3/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2'
%sequential_1/lstm_3/lstm_cell_3/add_1?
)sequential_1/lstm_3/lstm_cell_3/Sigmoid_2Sigmoid.sequential_1/lstm_3/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2+
)sequential_1/lstm_3/lstm_cell_3/Sigmoid_2?
&sequential_1/lstm_3/lstm_cell_3/Relu_1Relu)sequential_1/lstm_3/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2(
&sequential_1/lstm_3/lstm_cell_3/Relu_1?
%sequential_1/lstm_3/lstm_cell_3/mul_2Mul-sequential_1/lstm_3/lstm_cell_3/Sigmoid_2:y:04sequential_1/lstm_3/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2'
%sequential_1/lstm_3/lstm_cell_3/mul_2?
1sequential_1/lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   23
1sequential_1/lstm_3/TensorArrayV2_1/element_shape?
#sequential_1/lstm_3/TensorArrayV2_1TensorListReserve:sequential_1/lstm_3/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_1/lstm_3/TensorArrayV2_1v
sequential_1/lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_1/lstm_3/time?
,sequential_1/lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sequential_1/lstm_3/while/maximum_iterations?
&sequential_1/lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_1/lstm_3/while/loop_counter?
sequential_1/lstm_3/whileWhile/sequential_1/lstm_3/while/loop_counter:output:05sequential_1/lstm_3/while/maximum_iterations:output:0!sequential_1/lstm_3/time:output:0,sequential_1/lstm_3/TensorArrayV2_1:handle:0"sequential_1/lstm_3/zeros:output:0$sequential_1/lstm_3/zeros_1:output:0,sequential_1/lstm_3/strided_slice_1:output:0Ksequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_1_lstm_3_lstm_cell_3_matmul_readvariableop_resource@sequential_1_lstm_3_lstm_cell_3_matmul_1_readvariableop_resource?sequential_1_lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????(:?????????(: : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$sequential_1_lstm_3_while_body_75822*0
cond(R&
$sequential_1_lstm_3_while_cond_75821*K
output_shapes:
8: : : : :?????????(:?????????(: : : : : *
parallel_iterations 2
sequential_1/lstm_3/while?
Dsequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2F
Dsequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape?
6sequential_1/lstm_3/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_3/while:output:3Msequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????(*
element_dtype028
6sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack?
)sequential_1/lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)sequential_1/lstm_3/strided_slice_3/stack?
+sequential_1/lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_1/lstm_3/strided_slice_3/stack_1?
+sequential_1/lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_3/stack_2?
#sequential_1/lstm_3/strided_slice_3StridedSlice?sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_3/strided_slice_3/stack:output:04sequential_1/lstm_3/strided_slice_3/stack_1:output:04sequential_1/lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2%
#sequential_1/lstm_3/strided_slice_3?
$sequential_1/lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_1/lstm_3/transpose_1/perm?
sequential_1/lstm_3/transpose_1	Transpose?sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????(2!
sequential_1/lstm_3/transpose_1?
sequential_1/lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_1/lstm_3/runtime?
sequential_1/lstm_4/ShapeShape#sequential_1/lstm_3/transpose_1:y:0*
T0*
_output_shapes
:2
sequential_1/lstm_4/Shape?
'sequential_1/lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_1/lstm_4/strided_slice/stack?
)sequential_1/lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_4/strided_slice/stack_1?
)sequential_1/lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_4/strided_slice/stack_2?
!sequential_1/lstm_4/strided_sliceStridedSlice"sequential_1/lstm_4/Shape:output:00sequential_1/lstm_4/strided_slice/stack:output:02sequential_1/lstm_4/strided_slice/stack_1:output:02sequential_1/lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_1/lstm_4/strided_slice?
sequential_1/lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
sequential_1/lstm_4/zeros/mul/y?
sequential_1/lstm_4/zeros/mulMul*sequential_1/lstm_4/strided_slice:output:0(sequential_1/lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_4/zeros/mul?
 sequential_1/lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_1/lstm_4/zeros/Less/y?
sequential_1/lstm_4/zeros/LessLess!sequential_1/lstm_4/zeros/mul:z:0)sequential_1/lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_1/lstm_4/zeros/Less?
"sequential_1/lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"sequential_1/lstm_4/zeros/packed/1?
 sequential_1/lstm_4/zeros/packedPack*sequential_1/lstm_4/strided_slice:output:0+sequential_1/lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_1/lstm_4/zeros/packed?
sequential_1/lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_1/lstm_4/zeros/Const?
sequential_1/lstm_4/zerosFill)sequential_1/lstm_4/zeros/packed:output:0(sequential_1/lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
sequential_1/lstm_4/zeros?
!sequential_1/lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22#
!sequential_1/lstm_4/zeros_1/mul/y?
sequential_1/lstm_4/zeros_1/mulMul*sequential_1/lstm_4/strided_slice:output:0*sequential_1/lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_4/zeros_1/mul?
"sequential_1/lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_1/lstm_4/zeros_1/Less/y?
 sequential_1/lstm_4/zeros_1/LessLess#sequential_1/lstm_4/zeros_1/mul:z:0+sequential_1/lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_1/lstm_4/zeros_1/Less?
$sequential_1/lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22&
$sequential_1/lstm_4/zeros_1/packed/1?
"sequential_1/lstm_4/zeros_1/packedPack*sequential_1/lstm_4/strided_slice:output:0-sequential_1/lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/lstm_4/zeros_1/packed?
!sequential_1/lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_1/lstm_4/zeros_1/Const?
sequential_1/lstm_4/zeros_1Fill+sequential_1/lstm_4/zeros_1/packed:output:0*sequential_1/lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
sequential_1/lstm_4/zeros_1?
"sequential_1/lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_1/lstm_4/transpose/perm?
sequential_1/lstm_4/transpose	Transpose#sequential_1/lstm_3/transpose_1:y:0+sequential_1/lstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:?????????(2
sequential_1/lstm_4/transpose?
sequential_1/lstm_4/Shape_1Shape!sequential_1/lstm_4/transpose:y:0*
T0*
_output_shapes
:2
sequential_1/lstm_4/Shape_1?
)sequential_1/lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_4/strided_slice_1/stack?
+sequential_1/lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_4/strided_slice_1/stack_1?
+sequential_1/lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_4/strided_slice_1/stack_2?
#sequential_1/lstm_4/strided_slice_1StridedSlice$sequential_1/lstm_4/Shape_1:output:02sequential_1/lstm_4/strided_slice_1/stack:output:04sequential_1/lstm_4/strided_slice_1/stack_1:output:04sequential_1/lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_1/lstm_4/strided_slice_1?
/sequential_1/lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_1/lstm_4/TensorArrayV2/element_shape?
!sequential_1/lstm_4/TensorArrayV2TensorListReserve8sequential_1/lstm_4/TensorArrayV2/element_shape:output:0,sequential_1/lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_1/lstm_4/TensorArrayV2?
Isequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2K
Isequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape?
;sequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_4/transpose:y:0Rsequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensor?
)sequential_1/lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_4/strided_slice_2/stack?
+sequential_1/lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_4/strided_slice_2/stack_1?
+sequential_1/lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_4/strided_slice_2/stack_2?
#sequential_1/lstm_4/strided_slice_2StridedSlice!sequential_1/lstm_4/transpose:y:02sequential_1/lstm_4/strided_slice_2/stack:output:04sequential_1/lstm_4/strided_slice_2/stack_1:output:04sequential_1/lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2%
#sequential_1/lstm_4/strided_slice_2?
5sequential_1/lstm_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp>sequential_1_lstm_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype027
5sequential_1/lstm_4/lstm_cell_4/MatMul/ReadVariableOp?
&sequential_1/lstm_4/lstm_cell_4/MatMulMatMul,sequential_1/lstm_4/strided_slice_2:output:0=sequential_1/lstm_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential_1/lstm_4/lstm_cell_4/MatMul?
7sequential_1/lstm_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp@sequential_1_lstm_4_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype029
7sequential_1/lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp?
(sequential_1/lstm_4/lstm_cell_4/MatMul_1MatMul"sequential_1/lstm_4/zeros:output:0?sequential_1/lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_1/lstm_4/lstm_cell_4/MatMul_1?
#sequential_1/lstm_4/lstm_cell_4/addAddV20sequential_1/lstm_4/lstm_cell_4/MatMul:product:02sequential_1/lstm_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2%
#sequential_1/lstm_4/lstm_cell_4/add?
6sequential_1/lstm_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_lstm_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential_1/lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp?
'sequential_1/lstm_4/lstm_cell_4/BiasAddBiasAdd'sequential_1/lstm_4/lstm_cell_4/add:z:0>sequential_1/lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_1/lstm_4/lstm_cell_4/BiasAdd?
%sequential_1/lstm_4/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/lstm_4/lstm_cell_4/Const?
/sequential_1/lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_1/lstm_4/lstm_cell_4/split/split_dim?
%sequential_1/lstm_4/lstm_cell_4/splitSplit8sequential_1/lstm_4/lstm_cell_4/split/split_dim:output:00sequential_1/lstm_4/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2'
%sequential_1/lstm_4/lstm_cell_4/split?
'sequential_1/lstm_4/lstm_cell_4/SigmoidSigmoid.sequential_1/lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22)
'sequential_1/lstm_4/lstm_cell_4/Sigmoid?
)sequential_1/lstm_4/lstm_cell_4/Sigmoid_1Sigmoid.sequential_1/lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22+
)sequential_1/lstm_4/lstm_cell_4/Sigmoid_1?
#sequential_1/lstm_4/lstm_cell_4/mulMul-sequential_1/lstm_4/lstm_cell_4/Sigmoid_1:y:0$sequential_1/lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:?????????22%
#sequential_1/lstm_4/lstm_cell_4/mul?
$sequential_1/lstm_4/lstm_cell_4/ReluRelu.sequential_1/lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22&
$sequential_1/lstm_4/lstm_cell_4/Relu?
%sequential_1/lstm_4/lstm_cell_4/mul_1Mul+sequential_1/lstm_4/lstm_cell_4/Sigmoid:y:02sequential_1/lstm_4/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22'
%sequential_1/lstm_4/lstm_cell_4/mul_1?
%sequential_1/lstm_4/lstm_cell_4/add_1AddV2'sequential_1/lstm_4/lstm_cell_4/mul:z:0)sequential_1/lstm_4/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22'
%sequential_1/lstm_4/lstm_cell_4/add_1?
)sequential_1/lstm_4/lstm_cell_4/Sigmoid_2Sigmoid.sequential_1/lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22+
)sequential_1/lstm_4/lstm_cell_4/Sigmoid_2?
&sequential_1/lstm_4/lstm_cell_4/Relu_1Relu)sequential_1/lstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22(
&sequential_1/lstm_4/lstm_cell_4/Relu_1?
%sequential_1/lstm_4/lstm_cell_4/mul_2Mul-sequential_1/lstm_4/lstm_cell_4/Sigmoid_2:y:04sequential_1/lstm_4/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22'
%sequential_1/lstm_4/lstm_cell_4/mul_2?
1sequential_1/lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   23
1sequential_1/lstm_4/TensorArrayV2_1/element_shape?
#sequential_1/lstm_4/TensorArrayV2_1TensorListReserve:sequential_1/lstm_4/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_1/lstm_4/TensorArrayV2_1v
sequential_1/lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_1/lstm_4/time?
,sequential_1/lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sequential_1/lstm_4/while/maximum_iterations?
&sequential_1/lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_1/lstm_4/while/loop_counter?
sequential_1/lstm_4/whileWhile/sequential_1/lstm_4/while/loop_counter:output:05sequential_1/lstm_4/while/maximum_iterations:output:0!sequential_1/lstm_4/time:output:0,sequential_1/lstm_4/TensorArrayV2_1:handle:0"sequential_1/lstm_4/zeros:output:0$sequential_1/lstm_4/zeros_1:output:0,sequential_1/lstm_4/strided_slice_1:output:0Ksequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_1_lstm_4_lstm_cell_4_matmul_readvariableop_resource@sequential_1_lstm_4_lstm_cell_4_matmul_1_readvariableop_resource?sequential_1_lstm_4_lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$sequential_1_lstm_4_while_body_75971*0
cond(R&
$sequential_1_lstm_4_while_cond_75970*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
sequential_1/lstm_4/while?
Dsequential_1/lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2F
Dsequential_1/lstm_4/TensorArrayV2Stack/TensorListStack/element_shape?
6sequential_1/lstm_4/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_4/while:output:3Msequential_1/lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype028
6sequential_1/lstm_4/TensorArrayV2Stack/TensorListStack?
)sequential_1/lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)sequential_1/lstm_4/strided_slice_3/stack?
+sequential_1/lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_1/lstm_4/strided_slice_3/stack_1?
+sequential_1/lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_4/strided_slice_3/stack_2?
#sequential_1/lstm_4/strided_slice_3StridedSlice?sequential_1/lstm_4/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_4/strided_slice_3/stack:output:04sequential_1/lstm_4/strided_slice_3/stack_1:output:04sequential_1/lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2%
#sequential_1/lstm_4/strided_slice_3?
$sequential_1/lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_1/lstm_4/transpose_1/perm?
sequential_1/lstm_4/transpose_1	Transpose?sequential_1/lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22!
sequential_1/lstm_4/transpose_1?
sequential_1/lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_1/lstm_4/runtime?
sequential_1/lstm_5/ShapeShape#sequential_1/lstm_4/transpose_1:y:0*
T0*
_output_shapes
:2
sequential_1/lstm_5/Shape?
'sequential_1/lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_1/lstm_5/strided_slice/stack?
)sequential_1/lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_5/strided_slice/stack_1?
)sequential_1/lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_5/strided_slice/stack_2?
!sequential_1/lstm_5/strided_sliceStridedSlice"sequential_1/lstm_5/Shape:output:00sequential_1/lstm_5/strided_slice/stack:output:02sequential_1/lstm_5/strided_slice/stack_1:output:02sequential_1/lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_1/lstm_5/strided_slice?
sequential_1/lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22!
sequential_1/lstm_5/zeros/mul/y?
sequential_1/lstm_5/zeros/mulMul*sequential_1/lstm_5/strided_slice:output:0(sequential_1/lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_5/zeros/mul?
 sequential_1/lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_1/lstm_5/zeros/Less/y?
sequential_1/lstm_5/zeros/LessLess!sequential_1/lstm_5/zeros/mul:z:0)sequential_1/lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_1/lstm_5/zeros/Less?
"sequential_1/lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22$
"sequential_1/lstm_5/zeros/packed/1?
 sequential_1/lstm_5/zeros/packedPack*sequential_1/lstm_5/strided_slice:output:0+sequential_1/lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_1/lstm_5/zeros/packed?
sequential_1/lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_1/lstm_5/zeros/Const?
sequential_1/lstm_5/zerosFill)sequential_1/lstm_5/zeros/packed:output:0(sequential_1/lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
sequential_1/lstm_5/zeros?
!sequential_1/lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22#
!sequential_1/lstm_5/zeros_1/mul/y?
sequential_1/lstm_5/zeros_1/mulMul*sequential_1/lstm_5/strided_slice:output:0*sequential_1/lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_5/zeros_1/mul?
"sequential_1/lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_1/lstm_5/zeros_1/Less/y?
 sequential_1/lstm_5/zeros_1/LessLess#sequential_1/lstm_5/zeros_1/mul:z:0+sequential_1/lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_1/lstm_5/zeros_1/Less?
$sequential_1/lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22&
$sequential_1/lstm_5/zeros_1/packed/1?
"sequential_1/lstm_5/zeros_1/packedPack*sequential_1/lstm_5/strided_slice:output:0-sequential_1/lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/lstm_5/zeros_1/packed?
!sequential_1/lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_1/lstm_5/zeros_1/Const?
sequential_1/lstm_5/zeros_1Fill+sequential_1/lstm_5/zeros_1/packed:output:0*sequential_1/lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
sequential_1/lstm_5/zeros_1?
"sequential_1/lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_1/lstm_5/transpose/perm?
sequential_1/lstm_5/transpose	Transpose#sequential_1/lstm_4/transpose_1:y:0+sequential_1/lstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????22
sequential_1/lstm_5/transpose?
sequential_1/lstm_5/Shape_1Shape!sequential_1/lstm_5/transpose:y:0*
T0*
_output_shapes
:2
sequential_1/lstm_5/Shape_1?
)sequential_1/lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_5/strided_slice_1/stack?
+sequential_1/lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_5/strided_slice_1/stack_1?
+sequential_1/lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_5/strided_slice_1/stack_2?
#sequential_1/lstm_5/strided_slice_1StridedSlice$sequential_1/lstm_5/Shape_1:output:02sequential_1/lstm_5/strided_slice_1/stack:output:04sequential_1/lstm_5/strided_slice_1/stack_1:output:04sequential_1/lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_1/lstm_5/strided_slice_1?
/sequential_1/lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_1/lstm_5/TensorArrayV2/element_shape?
!sequential_1/lstm_5/TensorArrayV2TensorListReserve8sequential_1/lstm_5/TensorArrayV2/element_shape:output:0,sequential_1/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_1/lstm_5/TensorArrayV2?
Isequential_1/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2K
Isequential_1/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
;sequential_1/lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_5/transpose:y:0Rsequential_1/lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_1/lstm_5/TensorArrayUnstack/TensorListFromTensor?
)sequential_1/lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_5/strided_slice_2/stack?
+sequential_1/lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_5/strided_slice_2/stack_1?
+sequential_1/lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_5/strided_slice_2/stack_2?
#sequential_1/lstm_5/strided_slice_2StridedSlice!sequential_1/lstm_5/transpose:y:02sequential_1/lstm_5/strided_slice_2/stack:output:04sequential_1/lstm_5/strided_slice_2/stack_1:output:04sequential_1/lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2%
#sequential_1/lstm_5/strided_slice_2?
5sequential_1/lstm_5/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp>sequential_1_lstm_5_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	2?*
dtype027
5sequential_1/lstm_5/lstm_cell_5/MatMul/ReadVariableOp?
&sequential_1/lstm_5/lstm_cell_5/MatMulMatMul,sequential_1/lstm_5/strided_slice_2:output:0=sequential_1/lstm_5/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential_1/lstm_5/lstm_cell_5/MatMul?
7sequential_1/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp@sequential_1_lstm_5_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype029
7sequential_1/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp?
(sequential_1/lstm_5/lstm_cell_5/MatMul_1MatMul"sequential_1/lstm_5/zeros:output:0?sequential_1/lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_1/lstm_5/lstm_cell_5/MatMul_1?
#sequential_1/lstm_5/lstm_cell_5/addAddV20sequential_1/lstm_5/lstm_cell_5/MatMul:product:02sequential_1/lstm_5/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2%
#sequential_1/lstm_5/lstm_cell_5/add?
6sequential_1/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential_1/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp?
'sequential_1/lstm_5/lstm_cell_5/BiasAddBiasAdd'sequential_1/lstm_5/lstm_cell_5/add:z:0>sequential_1/lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_1/lstm_5/lstm_cell_5/BiasAdd?
%sequential_1/lstm_5/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/lstm_5/lstm_cell_5/Const?
/sequential_1/lstm_5/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_1/lstm_5/lstm_cell_5/split/split_dim?
%sequential_1/lstm_5/lstm_cell_5/splitSplit8sequential_1/lstm_5/lstm_cell_5/split/split_dim:output:00sequential_1/lstm_5/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2'
%sequential_1/lstm_5/lstm_cell_5/split?
'sequential_1/lstm_5/lstm_cell_5/SigmoidSigmoid.sequential_1/lstm_5/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22)
'sequential_1/lstm_5/lstm_cell_5/Sigmoid?
)sequential_1/lstm_5/lstm_cell_5/Sigmoid_1Sigmoid.sequential_1/lstm_5/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22+
)sequential_1/lstm_5/lstm_cell_5/Sigmoid_1?
#sequential_1/lstm_5/lstm_cell_5/mulMul-sequential_1/lstm_5/lstm_cell_5/Sigmoid_1:y:0$sequential_1/lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:?????????22%
#sequential_1/lstm_5/lstm_cell_5/mul?
$sequential_1/lstm_5/lstm_cell_5/ReluRelu.sequential_1/lstm_5/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22&
$sequential_1/lstm_5/lstm_cell_5/Relu?
%sequential_1/lstm_5/lstm_cell_5/mul_1Mul+sequential_1/lstm_5/lstm_cell_5/Sigmoid:y:02sequential_1/lstm_5/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22'
%sequential_1/lstm_5/lstm_cell_5/mul_1?
%sequential_1/lstm_5/lstm_cell_5/add_1AddV2'sequential_1/lstm_5/lstm_cell_5/mul:z:0)sequential_1/lstm_5/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22'
%sequential_1/lstm_5/lstm_cell_5/add_1?
)sequential_1/lstm_5/lstm_cell_5/Sigmoid_2Sigmoid.sequential_1/lstm_5/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22+
)sequential_1/lstm_5/lstm_cell_5/Sigmoid_2?
&sequential_1/lstm_5/lstm_cell_5/Relu_1Relu)sequential_1/lstm_5/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22(
&sequential_1/lstm_5/lstm_cell_5/Relu_1?
%sequential_1/lstm_5/lstm_cell_5/mul_2Mul-sequential_1/lstm_5/lstm_cell_5/Sigmoid_2:y:04sequential_1/lstm_5/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22'
%sequential_1/lstm_5/lstm_cell_5/mul_2?
1sequential_1/lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   23
1sequential_1/lstm_5/TensorArrayV2_1/element_shape?
#sequential_1/lstm_5/TensorArrayV2_1TensorListReserve:sequential_1/lstm_5/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_1/lstm_5/TensorArrayV2_1v
sequential_1/lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_1/lstm_5/time?
,sequential_1/lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sequential_1/lstm_5/while/maximum_iterations?
&sequential_1/lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_1/lstm_5/while/loop_counter?
sequential_1/lstm_5/whileWhile/sequential_1/lstm_5/while/loop_counter:output:05sequential_1/lstm_5/while/maximum_iterations:output:0!sequential_1/lstm_5/time:output:0,sequential_1/lstm_5/TensorArrayV2_1:handle:0"sequential_1/lstm_5/zeros:output:0$sequential_1/lstm_5/zeros_1:output:0,sequential_1/lstm_5/strided_slice_1:output:0Ksequential_1/lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_1_lstm_5_lstm_cell_5_matmul_readvariableop_resource@sequential_1_lstm_5_lstm_cell_5_matmul_1_readvariableop_resource?sequential_1_lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$sequential_1_lstm_5_while_body_76120*0
cond(R&
$sequential_1_lstm_5_while_cond_76119*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
sequential_1/lstm_5/while?
Dsequential_1/lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2F
Dsequential_1/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape?
6sequential_1/lstm_5/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_5/while:output:3Msequential_1/lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype028
6sequential_1/lstm_5/TensorArrayV2Stack/TensorListStack?
)sequential_1/lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)sequential_1/lstm_5/strided_slice_3/stack?
+sequential_1/lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_1/lstm_5/strided_slice_3/stack_1?
+sequential_1/lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_5/strided_slice_3/stack_2?
#sequential_1/lstm_5/strided_slice_3StridedSlice?sequential_1/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_5/strided_slice_3/stack:output:04sequential_1/lstm_5/strided_slice_3/stack_1:output:04sequential_1/lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2%
#sequential_1/lstm_5/strided_slice_3?
$sequential_1/lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_1/lstm_5/transpose_1/perm?
sequential_1/lstm_5/transpose_1	Transpose?sequential_1/lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22!
sequential_1/lstm_5/transpose_1?
sequential_1/lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_1/lstm_5/runtime?
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOp?
sequential_1/dense_1/MatMulMatMul,sequential_1/lstm_5/strided_slice_3:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
sequential_1/dense_1/MatMul?
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOp?
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
sequential_1/dense_1/BiasAdd?
IdentityIdentity%sequential_1/dense_1/BiasAdd:output:0^sequential_1/lstm_3/while^sequential_1/lstm_4/while^sequential_1/lstm_5/while*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::26
sequential_1/lstm_3/whilesequential_1/lstm_3/while26
sequential_1/lstm_4/whilesequential_1/lstm_4/while26
sequential_1/lstm_5/whilesequential_1/lstm_5/while:Y U
+
_output_shapes
:?????????
&
_user_specified_namelstm_3_input
?W
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_80380
inputs_0.
*lstm_cell_3_matmul_readvariableop_resource0
,lstm_cell_3_matmul_1_readvariableop_resource/
+lstm_cell_3_biasadd_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????(2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????(2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!lstm_cell_3/MatMul/ReadVariableOp?
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul?
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	(?*
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp?
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_1?
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add?
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAddh
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/Const|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dim?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
lstm_cell_3/split?
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid?
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid_1?
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mulz
lstm_cell_3/ReluRelulstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Relu?
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mul_1?
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/add_1?
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid_2y
lstm_cell_3/Relu_1Relulstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Relu_1?
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0 lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????(:?????????(: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_80295*
condR
while_cond_80294*K
output_shapes:
8: : : : :?????????(:?????????(: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????(*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????(2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :??????????????????(2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?V
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_78532

inputs.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????(2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/MatMul?
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp?
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/MatMul_1?
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/add?
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dim?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_4/split?
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid?
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid_1?
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Relu?
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mul_1?
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/add_1?
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Relu_1?
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_78447*
condR
while_cond_78446*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(:::2
whilewhile:S O
+
_output_shapes
:?????????(
 
_user_specified_nameinputs
?B
?
lstm_3_while_body_79327*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3)
%lstm_3_while_lstm_3_strided_slice_1_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0?
;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0>
:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5'
#lstm_3_while_lstm_3_strided_slice_1c
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor;
7lstm_3_while_lstm_cell_3_matmul_readvariableop_resource=
9lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource<
8lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource??
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0lstm_3/while/TensorArrayV2Read/TensorListGetItem?
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp?
lstm_3/while/lstm_cell_3/MatMulMatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_3/while/lstm_cell_3/MatMul?
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	(?*
dtype022
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp?
!lstm_3/while/lstm_cell_3/MatMul_1MatMullstm_3_while_placeholder_28lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_1?
lstm_3/while/lstm_cell_3/addAddV2)lstm_3/while/lstm_cell_3/MatMul:product:0+lstm_3/while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_3/while/lstm_cell_3/add?
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp?
 lstm_3/while/lstm_cell_3/BiasAddBiasAdd lstm_3/while/lstm_cell_3/add:z:07lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_3/while/lstm_cell_3/BiasAdd?
lstm_3/while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_3/while/lstm_cell_3/Const?
(lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_3/while/lstm_cell_3/split/split_dim?
lstm_3/while/lstm_cell_3/splitSplit1lstm_3/while/lstm_cell_3/split/split_dim:output:0)lstm_3/while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2 
lstm_3/while/lstm_cell_3/split?
 lstm_3/while/lstm_cell_3/SigmoidSigmoid'lstm_3/while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2"
 lstm_3/while/lstm_cell_3/Sigmoid?
"lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid'lstm_3/while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2$
"lstm_3/while/lstm_cell_3/Sigmoid_1?
lstm_3/while/lstm_cell_3/mulMul&lstm_3/while/lstm_cell_3/Sigmoid_1:y:0lstm_3_while_placeholder_3*
T0*'
_output_shapes
:?????????(2
lstm_3/while/lstm_cell_3/mul?
lstm_3/while/lstm_cell_3/ReluRelu'lstm_3/while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
lstm_3/while/lstm_cell_3/Relu?
lstm_3/while/lstm_cell_3/mul_1Mul$lstm_3/while/lstm_cell_3/Sigmoid:y:0+lstm_3/while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2 
lstm_3/while/lstm_cell_3/mul_1?
lstm_3/while/lstm_cell_3/add_1AddV2 lstm_3/while/lstm_cell_3/mul:z:0"lstm_3/while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2 
lstm_3/while/lstm_cell_3/add_1?
"lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid'lstm_3/while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2$
"lstm_3/while/lstm_cell_3/Sigmoid_2?
lstm_3/while/lstm_cell_3/Relu_1Relu"lstm_3/while/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2!
lstm_3/while/lstm_cell_3/Relu_1?
lstm_3/while/lstm_cell_3/mul_2Mul&lstm_3/while/lstm_cell_3/Sigmoid_2:y:0-lstm_3/while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2 
lstm_3/while/lstm_cell_3/mul_2?
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder"lstm_3/while/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_3/while/TensorArrayV2Write/TensorListSetItemj
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add/y?
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/addn
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add_1/y?
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/add_1s
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_3/while/Identity?
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_3/while/Identity_1u
lstm_3/while/Identity_2Identitylstm_3/while/add:z:0*
T0*
_output_shapes
: 2
lstm_3/while/Identity_2?
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_3/while/Identity_3?
lstm_3/while/Identity_4Identity"lstm_3/while/lstm_cell_3/mul_2:z:0*
T0*'
_output_shapes
:?????????(2
lstm_3/while/Identity_4?
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_3/while/Identity_5"7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"L
#lstm_3_while_lstm_3_strided_slice_1%lstm_3_while_lstm_3_strided_slice_1_0"v
8lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0"x
9lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0"t
7lstm_3_while_lstm_cell_3_matmul_readvariableop_resource9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0"?
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????(:?????????(: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
: 
?D
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_76680

inputs
lstm_cell_3_76598
lstm_cell_3_76600
lstm_cell_3_76602
identity??#lstm_cell_3/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????(2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????(2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_76598lstm_cell_3_76600lstm_cell_3_76602*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????(:?????????(:?????????(*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_762842%
#lstm_cell_3/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_76598lstm_cell_3_76600lstm_cell_3_76602*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????(:?????????(: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_76611*
condR
while_cond_76610*K
output_shapes:
8: : : : :?????????(:?????????(: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????(*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????(2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_3/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????(2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?9
?
while_body_81279
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_4_matmul_readvariableop_resource_08
4while_lstm_cell_4_matmul_1_readvariableop_resource_07
3while_lstm_cell_4_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_4_matmul_readvariableop_resource6
2while_lstm_cell_4_matmul_1_readvariableop_resource5
1while_lstm_cell_4_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????(*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	(?*
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/MatMul?
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp?
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/MatMul_1?
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/add?
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp?
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/BiasAddt
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const?
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_4/split?
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid?
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid_1?
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul?
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Relu?
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul_1?
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/add_1?
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid_2?
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Relu_1?
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_77504

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????2:?????????2:?????????2::::O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
?
?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_82247

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	(?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????(2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????(2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????(2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????(2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????(2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????(2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????(2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????(2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????(2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????(2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????(2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????(:?????????(::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????(
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????(
"
_user_specified_name
states/1
?D
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_77422

inputs
lstm_cell_4_77340
lstm_cell_4_77342
lstm_cell_4_77344
identity??#lstm_cell_4/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????(2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_77340lstm_cell_4_77342lstm_cell_4_77344*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_769272%
#lstm_cell_4/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_77340lstm_cell_4_77342lstm_cell_4_77344*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_77353*
condR
while_cond_77352*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_4/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????(:::2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????(
 
_user_specified_nameinputs
?$
?
while_body_77831
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_5_77855_0
while_lstm_cell_5_77857_0
while_lstm_cell_5_77859_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_5_77855
while_lstm_cell_5_77857
while_lstm_cell_5_77859??)while/lstm_cell_5/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_5_77855_0while_lstm_cell_5_77857_0while_lstm_cell_5_77859_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_775042+
)while/lstm_cell_5/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_5/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_5/StatefulPartitionedCall:output:1*^while/lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_5/StatefulPartitionedCall:output:2*^while/lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_5_77855while_lstm_cell_5_77855_0"4
while_lstm_cell_5_77857while_lstm_cell_5_77857_0"4
while_lstm_cell_5_77859while_lstm_cell_5_77859_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2V
)while/lstm_cell_5/StatefulPartitionedCall)while/lstm_cell_5/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?9
?
while_body_78447
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_4_matmul_readvariableop_resource_08
4while_lstm_cell_4_matmul_1_readvariableop_resource_07
3while_lstm_cell_4_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_4_matmul_readvariableop_resource6
2while_lstm_cell_4_matmul_1_readvariableop_resource5
1while_lstm_cell_4_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????(*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	(?*
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/MatMul?
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp?
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/MatMul_1?
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/add?
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp?
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/BiasAddt
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const?
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_4/split?
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid?
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid_1?
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul?
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Relu?
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul_1?
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/add_1?
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid_2?
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Relu_1?
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?B
?
lstm_3_while_body_79784*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3)
%lstm_3_while_lstm_3_strided_slice_1_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0?
;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0>
:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5'
#lstm_3_while_lstm_3_strided_slice_1c
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor;
7lstm_3_while_lstm_cell_3_matmul_readvariableop_resource=
9lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource<
8lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource??
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0lstm_3/while/TensorArrayV2Read/TensorListGetItem?
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp?
lstm_3/while/lstm_cell_3/MatMulMatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_3/while/lstm_cell_3/MatMul?
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	(?*
dtype022
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp?
!lstm_3/while/lstm_cell_3/MatMul_1MatMullstm_3_while_placeholder_28lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_3/while/lstm_cell_3/MatMul_1?
lstm_3/while/lstm_cell_3/addAddV2)lstm_3/while/lstm_cell_3/MatMul:product:0+lstm_3/while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_3/while/lstm_cell_3/add?
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp?
 lstm_3/while/lstm_cell_3/BiasAddBiasAdd lstm_3/while/lstm_cell_3/add:z:07lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_3/while/lstm_cell_3/BiasAdd?
lstm_3/while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_3/while/lstm_cell_3/Const?
(lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_3/while/lstm_cell_3/split/split_dim?
lstm_3/while/lstm_cell_3/splitSplit1lstm_3/while/lstm_cell_3/split/split_dim:output:0)lstm_3/while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2 
lstm_3/while/lstm_cell_3/split?
 lstm_3/while/lstm_cell_3/SigmoidSigmoid'lstm_3/while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2"
 lstm_3/while/lstm_cell_3/Sigmoid?
"lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid'lstm_3/while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2$
"lstm_3/while/lstm_cell_3/Sigmoid_1?
lstm_3/while/lstm_cell_3/mulMul&lstm_3/while/lstm_cell_3/Sigmoid_1:y:0lstm_3_while_placeholder_3*
T0*'
_output_shapes
:?????????(2
lstm_3/while/lstm_cell_3/mul?
lstm_3/while/lstm_cell_3/ReluRelu'lstm_3/while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
lstm_3/while/lstm_cell_3/Relu?
lstm_3/while/lstm_cell_3/mul_1Mul$lstm_3/while/lstm_cell_3/Sigmoid:y:0+lstm_3/while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2 
lstm_3/while/lstm_cell_3/mul_1?
lstm_3/while/lstm_cell_3/add_1AddV2 lstm_3/while/lstm_cell_3/mul:z:0"lstm_3/while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2 
lstm_3/while/lstm_cell_3/add_1?
"lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid'lstm_3/while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2$
"lstm_3/while/lstm_cell_3/Sigmoid_2?
lstm_3/while/lstm_cell_3/Relu_1Relu"lstm_3/while/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2!
lstm_3/while/lstm_cell_3/Relu_1?
lstm_3/while/lstm_cell_3/mul_2Mul&lstm_3/while/lstm_cell_3/Sigmoid_2:y:0-lstm_3/while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2 
lstm_3/while/lstm_cell_3/mul_2?
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder"lstm_3/while/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_3/while/TensorArrayV2Write/TensorListSetItemj
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add/y?
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/addn
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add_1/y?
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/add_1s
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_3/while/Identity?
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_3/while/Identity_1u
lstm_3/while/Identity_2Identitylstm_3/while/add:z:0*
T0*
_output_shapes
: 2
lstm_3/while/Identity_2?
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_3/while/Identity_3?
lstm_3/while/Identity_4Identity"lstm_3/while/lstm_cell_3/mul_2:z:0*
T0*'
_output_shapes
:?????????(2
lstm_3/while/Identity_4?
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_3/while/Identity_5"7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"L
#lstm_3_while_lstm_3_strided_slice_1%lstm_3_while_lstm_3_strided_slice_1_0"v
8lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0"x
9lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0"t
7lstm_3_while_lstm_cell_3_matmul_readvariableop_resource9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0"?
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????(:?????????(: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_79077
lstm_3_input
lstm_3_78373
lstm_3_78375
lstm_3_78377
lstm_4_78708
lstm_4_78710
lstm_4_78712
lstm_5_79043
lstm_5_79045
lstm_5_79047
dense_1_79071
dense_1_79073
identity??dense_1/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?lstm_4/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCalllstm_3_inputlstm_3_78373lstm_3_78375lstm_3_78377*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????(*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_781972 
lstm_3/StatefulPartitionedCall?
lstm_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0lstm_4_78708lstm_4_78710lstm_4_78712*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_785322 
lstm_4/StatefulPartitionedCall?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0lstm_5_79043lstm_5_79045lstm_5_79047*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_788672 
lstm_5/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_1_79071dense_1_79073*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_790602!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namelstm_3_input
?
?
&__inference_lstm_3_layer_call_fn_80544
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????(*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_766802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????(2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?W
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_78867

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_78782*
condR
while_cond_78781*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????2:::2
whilewhile:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
while_cond_81278
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_81278___redundant_placeholder03
/while_while_cond_81278___redundant_placeholder13
/while_while_cond_81278___redundant_placeholder23
/while_while_cond_81278___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?	
?
,__inference_sequential_1_layer_call_fn_80227

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_791972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
lstm_3_while_cond_79326*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3,
(lstm_3_while_less_lstm_3_strided_slice_1A
=lstm_3_while_lstm_3_while_cond_79326___redundant_placeholder0A
=lstm_3_while_lstm_3_while_cond_79326___redundant_placeholder1A
=lstm_3_while_lstm_3_while_cond_79326___redundant_placeholder2A
=lstm_3_while_lstm_3_while_cond_79326___redundant_placeholder3
lstm_3_while_identity
?
lstm_3/while/LessLesslstm_3_while_placeholder(lstm_3_while_less_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
lstm_3/while/Lessr
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_3/while/Identity"7
lstm_3_while_identitylstm_3/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????(:?????????(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
:
??
?
!__inference__traced_restore_82799
file_prefix#
assignvariableop_dense_1_kernel#
assignvariableop_1_dense_1_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate0
,assignvariableop_7_lstm_3_lstm_cell_3_kernel:
6assignvariableop_8_lstm_3_lstm_cell_3_recurrent_kernel.
*assignvariableop_9_lstm_3_lstm_cell_3_bias1
-assignvariableop_10_lstm_4_lstm_cell_4_kernel;
7assignvariableop_11_lstm_4_lstm_cell_4_recurrent_kernel/
+assignvariableop_12_lstm_4_lstm_cell_4_bias1
-assignvariableop_13_lstm_5_lstm_cell_5_kernel;
7assignvariableop_14_lstm_5_lstm_cell_5_recurrent_kernel/
+assignvariableop_15_lstm_5_lstm_cell_5_bias
assignvariableop_16_total
assignvariableop_17_count
assignvariableop_18_total_1
assignvariableop_19_count_1-
)assignvariableop_20_adam_dense_1_kernel_m+
'assignvariableop_21_adam_dense_1_bias_m8
4assignvariableop_22_adam_lstm_3_lstm_cell_3_kernel_mB
>assignvariableop_23_adam_lstm_3_lstm_cell_3_recurrent_kernel_m6
2assignvariableop_24_adam_lstm_3_lstm_cell_3_bias_m8
4assignvariableop_25_adam_lstm_4_lstm_cell_4_kernel_mB
>assignvariableop_26_adam_lstm_4_lstm_cell_4_recurrent_kernel_m6
2assignvariableop_27_adam_lstm_4_lstm_cell_4_bias_m8
4assignvariableop_28_adam_lstm_5_lstm_cell_5_kernel_mB
>assignvariableop_29_adam_lstm_5_lstm_cell_5_recurrent_kernel_m6
2assignvariableop_30_adam_lstm_5_lstm_cell_5_bias_m-
)assignvariableop_31_adam_dense_1_kernel_v+
'assignvariableop_32_adam_dense_1_bias_v8
4assignvariableop_33_adam_lstm_3_lstm_cell_3_kernel_vB
>assignvariableop_34_adam_lstm_3_lstm_cell_3_recurrent_kernel_v6
2assignvariableop_35_adam_lstm_3_lstm_cell_3_bias_v8
4assignvariableop_36_adam_lstm_4_lstm_cell_4_kernel_vB
>assignvariableop_37_adam_lstm_4_lstm_cell_4_recurrent_kernel_v6
2assignvariableop_38_adam_lstm_4_lstm_cell_4_bias_v8
4assignvariableop_39_adam_lstm_5_lstm_cell_5_kernel_vB
>assignvariableop_40_adam_lstm_5_lstm_cell_5_recurrent_kernel_v6
2assignvariableop_41_adam_lstm_5_lstm_cell_5_bias_v
identity_43??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*?
value?B?+B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_3_lstm_cell_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp6assignvariableop_8_lstm_3_lstm_cell_3_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_lstm_3_lstm_cell_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp-assignvariableop_10_lstm_4_lstm_cell_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp7assignvariableop_11_lstm_4_lstm_cell_4_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp+assignvariableop_12_lstm_4_lstm_cell_4_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp-assignvariableop_13_lstm_5_lstm_cell_5_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp7assignvariableop_14_lstm_5_lstm_cell_5_recurrent_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp+assignvariableop_15_lstm_5_lstm_cell_5_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_1_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_1_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_lstm_3_lstm_cell_3_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_lstm_3_lstm_cell_3_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_lstm_3_lstm_cell_3_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_lstm_4_lstm_cell_4_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp>assignvariableop_26_adam_lstm_4_lstm_cell_4_recurrent_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_lstm_4_lstm_cell_4_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_lstm_5_lstm_cell_5_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_lstm_5_lstm_cell_5_recurrent_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_lstm_5_lstm_cell_5_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_lstm_3_lstm_cell_3_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp>assignvariableop_34_adam_lstm_3_lstm_cell_3_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adam_lstm_3_lstm_cell_3_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp4assignvariableop_36_adam_lstm_4_lstm_cell_4_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp>assignvariableop_37_adam_lstm_4_lstm_cell_4_recurrent_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adam_lstm_4_lstm_cell_4_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adam_lstm_5_lstm_cell_5_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp>assignvariableop_40_adam_lstm_5_lstm_cell_5_recurrent_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp2assignvariableop_41_adam_lstm_5_lstm_cell_5_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_419
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_42?
Identity_43IdentityIdentity_42:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_43"#
identity_43Identity_43:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
&__inference_lstm_5_layer_call_fn_82195

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_790202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?Z
?
__inference__traced_save_82663
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_3_lstm_cell_3_kernel_read_readvariableopB
>savev2_lstm_3_lstm_cell_3_recurrent_kernel_read_readvariableop6
2savev2_lstm_3_lstm_cell_3_bias_read_readvariableop8
4savev2_lstm_4_lstm_cell_4_kernel_read_readvariableopB
>savev2_lstm_4_lstm_cell_4_recurrent_kernel_read_readvariableop6
2savev2_lstm_4_lstm_cell_4_bias_read_readvariableop8
4savev2_lstm_5_lstm_cell_5_kernel_read_readvariableopB
>savev2_lstm_5_lstm_cell_5_recurrent_kernel_read_readvariableop6
2savev2_lstm_5_lstm_cell_5_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_3_kernel_m_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_3_bias_m_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_4_kernel_m_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_4_bias_m_read_readvariableop?
;savev2_adam_lstm_5_lstm_cell_5_kernel_m_read_readvariableopI
Esavev2_adam_lstm_5_lstm_cell_5_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_5_lstm_cell_5_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_3_kernel_v_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_3_bias_v_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_4_kernel_v_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_4_bias_v_read_readvariableop?
;savev2_adam_lstm_5_lstm_cell_5_kernel_v_read_readvariableopI
Esavev2_adam_lstm_5_lstm_cell_5_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_5_lstm_cell_5_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_18e759e4912f4f769e8c1ead8412ea8f/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*?
value?B?+B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_3_lstm_cell_3_kernel_read_readvariableop>savev2_lstm_3_lstm_cell_3_recurrent_kernel_read_readvariableop2savev2_lstm_3_lstm_cell_3_bias_read_readvariableop4savev2_lstm_4_lstm_cell_4_kernel_read_readvariableop>savev2_lstm_4_lstm_cell_4_recurrent_kernel_read_readvariableop2savev2_lstm_4_lstm_cell_4_bias_read_readvariableop4savev2_lstm_5_lstm_cell_5_kernel_read_readvariableop>savev2_lstm_5_lstm_cell_5_recurrent_kernel_read_readvariableop2savev2_lstm_5_lstm_cell_5_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop;savev2_adam_lstm_3_lstm_cell_3_kernel_m_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_3_lstm_cell_3_bias_m_read_readvariableop;savev2_adam_lstm_4_lstm_cell_4_kernel_m_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_4_lstm_cell_4_bias_m_read_readvariableop;savev2_adam_lstm_5_lstm_cell_5_kernel_m_read_readvariableopEsavev2_adam_lstm_5_lstm_cell_5_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_5_lstm_cell_5_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop;savev2_adam_lstm_3_lstm_cell_3_kernel_v_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_3_lstm_cell_3_bias_v_read_readvariableop;savev2_adam_lstm_4_lstm_cell_4_kernel_v_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_4_lstm_cell_4_bias_v_read_readvariableop;savev2_adam_lstm_5_lstm_cell_5_kernel_v_read_readvariableopEsavev2_adam_lstm_5_lstm_cell_5_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_5_lstm_cell_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :2(:(: : : : : :	?:	(?:?:	(?:	2?:?:	2?:	2?:?: : : : :2(:(:	?:	(?:?:	(?:	2?:?:	2?:	2?:?:2(:(:	?:	(?:?:	(?:	2?:?:	2?:	2?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2(: 

_output_shapes
:(:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:%	!

_output_shapes
:	(?:!


_output_shapes	
:?:%!

_output_shapes
:	(?:%!

_output_shapes
:	2?:!

_output_shapes	
:?:%!

_output_shapes
:	2?:%!

_output_shapes
:	2?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2(: 

_output_shapes
:(:%!

_output_shapes
:	?:%!

_output_shapes
:	(?:!

_output_shapes	
:?:%!

_output_shapes
:	(?:%!

_output_shapes
:	2?:!

_output_shapes	
:?:%!

_output_shapes
:	2?:%!

_output_shapes
:	2?:!

_output_shapes	
:?:$  

_output_shapes

:2(: !

_output_shapes
:(:%"!

_output_shapes
:	?:%#!

_output_shapes
:	(?:!$

_output_shapes	
:?:%%!

_output_shapes
:	(?:%&!

_output_shapes
:	2?:!'

_output_shapes	
:?:%(!

_output_shapes
:	2?:%)!

_output_shapes
:	2?:!*

_output_shapes	
:?:+

_output_shapes
: 
?
?
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_76894

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????(:?????????2:?????????2::::O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
?
?
while_cond_78111
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_78111___redundant_placeholder03
/while_while_cond_78111___redundant_placeholder13
/while_while_cond_78111___redundant_placeholder23
/while_while_cond_78111___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????(:?????????(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
:
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_79197

inputs
lstm_3_79170
lstm_3_79172
lstm_3_79174
lstm_4_79177
lstm_4_79179
lstm_4_79181
lstm_5_79184
lstm_5_79186
lstm_5_79188
dense_1_79191
dense_1_79193
identity??dense_1/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?lstm_4/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCallinputslstm_3_79170lstm_3_79172lstm_3_79174*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????(*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_783502 
lstm_3/StatefulPartitionedCall?
lstm_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0lstm_4_79177lstm_4_79179lstm_4_79181*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_786852 
lstm_4/StatefulPartitionedCall?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0lstm_5_79184lstm_5_79186lstm_5_79188*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_790202 
lstm_5/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_1_79191dense_1_79193*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_790602!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
,__inference_sequential_1_layer_call_fn_79222
lstm_3_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_791972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namelstm_3_input
?R
?
$sequential_1_lstm_3_while_body_75822D
@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counterJ
Fsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations)
%sequential_1_lstm_3_while_placeholder+
'sequential_1_lstm_3_while_placeholder_1+
'sequential_1_lstm_3_while_placeholder_2+
'sequential_1_lstm_3_while_placeholder_3C
?sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1_0
{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0J
Fsequential_1_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0L
Hsequential_1_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0K
Gsequential_1_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0&
"sequential_1_lstm_3_while_identity(
$sequential_1_lstm_3_while_identity_1(
$sequential_1_lstm_3_while_identity_2(
$sequential_1_lstm_3_while_identity_3(
$sequential_1_lstm_3_while_identity_4(
$sequential_1_lstm_3_while_identity_5A
=sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1}
ysequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensorH
Dsequential_1_lstm_3_while_lstm_cell_3_matmul_readvariableop_resourceJ
Fsequential_1_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resourceI
Esequential_1_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource??
Ksequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2M
Ksequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=sequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_3_while_placeholderTsequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02?
=sequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem?
;sequential_1/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOpFsequential_1_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02=
;sequential_1/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp?
,sequential_1/lstm_3/while/lstm_cell_3/MatMulMatMulDsequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_1/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,sequential_1/lstm_3/while/lstm_cell_3/MatMul?
=sequential_1/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOpHsequential_1_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	(?*
dtype02?
=sequential_1/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp?
.sequential_1/lstm_3/while/lstm_cell_3/MatMul_1MatMul'sequential_1_lstm_3_while_placeholder_2Esequential_1/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_1/lstm_3/while/lstm_cell_3/MatMul_1?
)sequential_1/lstm_3/while/lstm_cell_3/addAddV26sequential_1/lstm_3/while/lstm_cell_3/MatMul:product:08sequential_1/lstm_3/while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2+
)sequential_1/lstm_3/while/lstm_cell_3/add?
<sequential_1/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOpGsequential_1_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02>
<sequential_1/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp?
-sequential_1/lstm_3/while/lstm_cell_3/BiasAddBiasAdd-sequential_1/lstm_3/while/lstm_cell_3/add:z:0Dsequential_1/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-sequential_1/lstm_3/while/lstm_cell_3/BiasAdd?
+sequential_1/lstm_3/while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+sequential_1/lstm_3/while/lstm_cell_3/Const?
5sequential_1/lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_1/lstm_3/while/lstm_cell_3/split/split_dim?
+sequential_1/lstm_3/while/lstm_cell_3/splitSplit>sequential_1/lstm_3/while/lstm_cell_3/split/split_dim:output:06sequential_1/lstm_3/while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2-
+sequential_1/lstm_3/while/lstm_cell_3/split?
-sequential_1/lstm_3/while/lstm_cell_3/SigmoidSigmoid4sequential_1/lstm_3/while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2/
-sequential_1/lstm_3/while/lstm_cell_3/Sigmoid?
/sequential_1/lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid4sequential_1/lstm_3/while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(21
/sequential_1/lstm_3/while/lstm_cell_3/Sigmoid_1?
)sequential_1/lstm_3/while/lstm_cell_3/mulMul3sequential_1/lstm_3/while/lstm_cell_3/Sigmoid_1:y:0'sequential_1_lstm_3_while_placeholder_3*
T0*'
_output_shapes
:?????????(2+
)sequential_1/lstm_3/while/lstm_cell_3/mul?
*sequential_1/lstm_3/while/lstm_cell_3/ReluRelu4sequential_1/lstm_3/while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2,
*sequential_1/lstm_3/while/lstm_cell_3/Relu?
+sequential_1/lstm_3/while/lstm_cell_3/mul_1Mul1sequential_1/lstm_3/while/lstm_cell_3/Sigmoid:y:08sequential_1/lstm_3/while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2-
+sequential_1/lstm_3/while/lstm_cell_3/mul_1?
+sequential_1/lstm_3/while/lstm_cell_3/add_1AddV2-sequential_1/lstm_3/while/lstm_cell_3/mul:z:0/sequential_1/lstm_3/while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2-
+sequential_1/lstm_3/while/lstm_cell_3/add_1?
/sequential_1/lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid4sequential_1/lstm_3/while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(21
/sequential_1/lstm_3/while/lstm_cell_3/Sigmoid_2?
,sequential_1/lstm_3/while/lstm_cell_3/Relu_1Relu/sequential_1/lstm_3/while/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2.
,sequential_1/lstm_3/while/lstm_cell_3/Relu_1?
+sequential_1/lstm_3/while/lstm_cell_3/mul_2Mul3sequential_1/lstm_3/while/lstm_cell_3/Sigmoid_2:y:0:sequential_1/lstm_3/while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2-
+sequential_1/lstm_3/while/lstm_cell_3/mul_2?
>sequential_1/lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_3_while_placeholder_1%sequential_1_lstm_3_while_placeholder/sequential_1/lstm_3/while/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_1/lstm_3/while/TensorArrayV2Write/TensorListSetItem?
sequential_1/lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_1/lstm_3/while/add/y?
sequential_1/lstm_3/while/addAddV2%sequential_1_lstm_3_while_placeholder(sequential_1/lstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_3/while/add?
!sequential_1/lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_1/lstm_3/while/add_1/y?
sequential_1/lstm_3/while/add_1AddV2@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counter*sequential_1/lstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_3/while/add_1?
"sequential_1/lstm_3/while/IdentityIdentity#sequential_1/lstm_3/while/add_1:z:0*
T0*
_output_shapes
: 2$
"sequential_1/lstm_3/while/Identity?
$sequential_1/lstm_3/while/Identity_1IdentityFsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations*
T0*
_output_shapes
: 2&
$sequential_1/lstm_3/while/Identity_1?
$sequential_1/lstm_3/while/Identity_2Identity!sequential_1/lstm_3/while/add:z:0*
T0*
_output_shapes
: 2&
$sequential_1/lstm_3/while/Identity_2?
$sequential_1/lstm_3/while/Identity_3IdentityNsequential_1/lstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2&
$sequential_1/lstm_3/while/Identity_3?
$sequential_1/lstm_3/while/Identity_4Identity/sequential_1/lstm_3/while/lstm_cell_3/mul_2:z:0*
T0*'
_output_shapes
:?????????(2&
$sequential_1/lstm_3/while/Identity_4?
$sequential_1/lstm_3/while/Identity_5Identity/sequential_1/lstm_3/while/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2&
$sequential_1/lstm_3/while/Identity_5"Q
"sequential_1_lstm_3_while_identity+sequential_1/lstm_3/while/Identity:output:0"U
$sequential_1_lstm_3_while_identity_1-sequential_1/lstm_3/while/Identity_1:output:0"U
$sequential_1_lstm_3_while_identity_2-sequential_1/lstm_3/while/Identity_2:output:0"U
$sequential_1_lstm_3_while_identity_3-sequential_1/lstm_3/while/Identity_3:output:0"U
$sequential_1_lstm_3_while_identity_4-sequential_1/lstm_3/while/Identity_4:output:0"U
$sequential_1_lstm_3_while_identity_5-sequential_1/lstm_3/while/Identity_5:output:0"?
Esequential_1_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resourceGsequential_1_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0"?
Fsequential_1_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resourceHsequential_1_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0"?
Dsequential_1_lstm_3_while_lstm_cell_3_matmul_readvariableop_resourceFsequential_1_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0"?
=sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1?sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1_0"?
ysequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????(:?????????(: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_81759
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_81759___redundant_placeholder03
/while_while_cond_81759___redundant_placeholder13
/while_while_cond_81759___redundant_placeholder23
/while_while_cond_81759___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?9
?
while_body_78600
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_4_matmul_readvariableop_resource_08
4while_lstm_cell_4_matmul_1_readvariableop_resource_07
3while_lstm_cell_4_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_4_matmul_readvariableop_resource6
2while_lstm_cell_4_matmul_1_readvariableop_resource5
1while_lstm_cell_4_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????(*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	(?*
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/MatMul?
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp?
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/MatMul_1?
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/add?
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp?
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/BiasAddt
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const?
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_4/split?
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid?
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid_1?
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul?
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Relu?
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul_1?
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/add_1?
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid_2?
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Relu_1?
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_lstm_3_layer_call_fn_80872

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????(*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_781972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?9
?
while_body_81607
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?V
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_78197

inputs.
*lstm_cell_3_matmul_readvariableop_resource0
,lstm_cell_3_matmul_1_readvariableop_resource/
+lstm_cell_3_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????(2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????(2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!lstm_cell_3/MatMul/ReadVariableOp?
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul?
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	(?*
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp?
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_1?
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add?
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAddh
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/Const|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dim?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
lstm_cell_3/split?
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid?
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid_1?
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mulz
lstm_cell_3/ReluRelulstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Relu?
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mul_1?
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/add_1?
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid_2y
lstm_cell_3/Relu_1Relulstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Relu_1?
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0 lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????(:?????????(: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_78112*
condR
while_cond_78111*K
output_shapes:
8: : : : :?????????(:?????????(: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????(*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????(2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?9
?
while_body_78782
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_78264
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_78264___redundant_placeholder03
/while_while_cond_78264___redundant_placeholder13
/while_while_cond_78264___redundant_placeholder23
/while_while_cond_78264___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????(:?????????(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
:
?
?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_76317

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	(?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????(2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????(2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????(2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????(2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????(2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????(2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????(2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????(2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????(2

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????(2

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????(2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????(:?????????(::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????(
 
_user_specified_namestates:OK
'
_output_shapes
:?????????(
 
_user_specified_namestates
?	
?
lstm_4_while_cond_79932*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3,
(lstm_4_while_less_lstm_4_strided_slice_1A
=lstm_4_while_lstm_4_while_cond_79932___redundant_placeholder0A
=lstm_4_while_lstm_4_while_cond_79932___redundant_placeholder1A
=lstm_4_while_lstm_4_while_cond_79932___redundant_placeholder2A
=lstm_4_while_lstm_4_while_cond_79932___redundant_placeholder3
lstm_4_while_identity
?
lstm_4/while/LessLesslstm_4_while_placeholder(lstm_4_while_less_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
lstm_4/while/Lessr
lstm_4/while/IdentityIdentitylstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_4/while/Identity"7
lstm_4_while_identitylstm_4/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?9
?
while_body_81935
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_76610
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_76610___redundant_placeholder03
/while_while_cond_76610___redundant_placeholder13
/while_while_cond_76610___redundant_placeholder23
/while_while_cond_76610___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????(:?????????(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
:
?
?
&__inference_lstm_5_layer_call_fn_81867
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_780322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????2
"
_user_specified_name
inputs/0
?
|
'__inference_dense_1_layer_call_fn_82214

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_790602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?V
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_78685

inputs.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????(2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/MatMul?
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp?
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/MatMul_1?
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/add?
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dim?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_4/split?
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid?
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid_1?
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Relu?
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mul_1?
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/add_1?
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Relu_1?
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_78600*
condR
while_cond_78599*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(:::2
whilewhile:S O
+
_output_shapes
:?????????(
 
_user_specified_nameinputs
?	
?
lstm_5_while_cond_80081*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1A
=lstm_5_while_lstm_5_while_cond_80081___redundant_placeholder0A
=lstm_5_while_lstm_5_while_cond_80081___redundant_placeholder1A
=lstm_5_while_lstm_5_while_cond_80081___redundant_placeholder2A
=lstm_5_while_lstm_5_while_cond_80081___redundant_placeholder3
lstm_5_while_identity
?
lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2
lstm_5/while/Lessr
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_5/while/Identity"7
lstm_5_while_identitylstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?	
?
lstm_5_while_cond_79624*
&lstm_5_while_lstm_5_while_loop_counter0
,lstm_5_while_lstm_5_while_maximum_iterations
lstm_5_while_placeholder
lstm_5_while_placeholder_1
lstm_5_while_placeholder_2
lstm_5_while_placeholder_3,
(lstm_5_while_less_lstm_5_strided_slice_1A
=lstm_5_while_lstm_5_while_cond_79624___redundant_placeholder0A
=lstm_5_while_lstm_5_while_cond_79624___redundant_placeholder1A
=lstm_5_while_lstm_5_while_cond_79624___redundant_placeholder2A
=lstm_5_while_lstm_5_while_cond_79624___redundant_placeholder3
lstm_5_while_identity
?
lstm_5/while/LessLesslstm_5_while_placeholder(lstm_5_while_less_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2
lstm_5/while/Lessr
lstm_5/while/IdentityIdentitylstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_5/while/Identity"7
lstm_5_while_identitylstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?B
?
lstm_4_while_body_79476*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3)
%lstm_4_while_lstm_4_strided_slice_1_0e
alstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0=
9lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0?
;lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0>
:lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0
lstm_4_while_identity
lstm_4_while_identity_1
lstm_4_while_identity_2
lstm_4_while_identity_3
lstm_4_while_identity_4
lstm_4_while_identity_5'
#lstm_4_while_lstm_4_strided_slice_1c
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor;
7lstm_4_while_lstm_cell_4_matmul_readvariableop_resource=
9lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource<
8lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource??
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2@
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0lstm_4_while_placeholderGlstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????(*
element_dtype022
0lstm_4/while/TensorArrayV2Read/TensorListGetItem?
.lstm_4/while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp9lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	(?*
dtype020
.lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp?
lstm_4/while/lstm_cell_4/MatMulMatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
lstm_4/while/lstm_cell_4/MatMul?
0lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp;lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype022
0lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp?
!lstm_4/while/lstm_cell_4/MatMul_1MatMullstm_4_while_placeholder_28lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_4/while/lstm_cell_4/MatMul_1?
lstm_4/while/lstm_cell_4/addAddV2)lstm_4/while/lstm_cell_4/MatMul:product:0+lstm_4/while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_4/while/lstm_cell_4/add?
/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp:lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp?
 lstm_4/while/lstm_cell_4/BiasAddBiasAdd lstm_4/while/lstm_cell_4/add:z:07lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 lstm_4/while/lstm_cell_4/BiasAdd?
lstm_4/while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm_4/while/lstm_cell_4/Const?
(lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_4/while/lstm_cell_4/split/split_dim?
lstm_4/while/lstm_cell_4/splitSplit1lstm_4/while/lstm_cell_4/split/split_dim:output:0)lstm_4/while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2 
lstm_4/while/lstm_cell_4/split?
 lstm_4/while/lstm_cell_4/SigmoidSigmoid'lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22"
 lstm_4/while/lstm_cell_4/Sigmoid?
"lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid'lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22$
"lstm_4/while/lstm_cell_4/Sigmoid_1?
lstm_4/while/lstm_cell_4/mulMul&lstm_4/while/lstm_cell_4/Sigmoid_1:y:0lstm_4_while_placeholder_3*
T0*'
_output_shapes
:?????????22
lstm_4/while/lstm_cell_4/mul?
lstm_4/while/lstm_cell_4/ReluRelu'lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_4/while/lstm_cell_4/Relu?
lstm_4/while/lstm_cell_4/mul_1Mul$lstm_4/while/lstm_cell_4/Sigmoid:y:0+lstm_4/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22 
lstm_4/while/lstm_cell_4/mul_1?
lstm_4/while/lstm_cell_4/add_1AddV2 lstm_4/while/lstm_cell_4/mul:z:0"lstm_4/while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22 
lstm_4/while/lstm_cell_4/add_1?
"lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid'lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22$
"lstm_4/while/lstm_cell_4/Sigmoid_2?
lstm_4/while/lstm_cell_4/Relu_1Relu"lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22!
lstm_4/while/lstm_cell_4/Relu_1?
lstm_4/while/lstm_cell_4/mul_2Mul&lstm_4/while/lstm_cell_4/Sigmoid_2:y:0-lstm_4/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22 
lstm_4/while/lstm_cell_4/mul_2?
1lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_4_while_placeholder_1lstm_4_while_placeholder"lstm_4/while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_4/while/TensorArrayV2Write/TensorListSetItemj
lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_4/while/add/y?
lstm_4/while/addAddV2lstm_4_while_placeholderlstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_4/while/addn
lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_4/while/add_1/y?
lstm_4/while/add_1AddV2&lstm_4_while_lstm_4_while_loop_counterlstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_4/while/add_1s
lstm_4/while/IdentityIdentitylstm_4/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm_4/while/Identity?
lstm_4/while/Identity_1Identity,lstm_4_while_lstm_4_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm_4/while/Identity_1u
lstm_4/while/Identity_2Identitylstm_4/while/add:z:0*
T0*
_output_shapes
: 2
lstm_4/while/Identity_2?
lstm_4/while/Identity_3IdentityAlstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm_4/while/Identity_3?
lstm_4/while/Identity_4Identity"lstm_4/while/lstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????22
lstm_4/while/Identity_4?
lstm_4/while/Identity_5Identity"lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_4/while/Identity_5"7
lstm_4_while_identitylstm_4/while/Identity:output:0";
lstm_4_while_identity_1 lstm_4/while/Identity_1:output:0";
lstm_4_while_identity_2 lstm_4/while/Identity_2:output:0";
lstm_4_while_identity_3 lstm_4/while/Identity_3:output:0";
lstm_4_while_identity_4 lstm_4/while/Identity_4:output:0";
lstm_4_while_identity_5 lstm_4/while/Identity_5:output:0"L
#lstm_4_while_lstm_4_strided_slice_1%lstm_4_while_lstm_4_strided_slice_1_0"v
8lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource:lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0"x
9lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource;lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0"t
7lstm_4_while_lstm_cell_4_matmul_readvariableop_resource9lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0"?
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensoralstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_lstm_4_layer_call_fn_81539

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_786852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????(
 
_user_specified_nameinputs
?D
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_77290

inputs
lstm_cell_4_77208
lstm_cell_4_77210
lstm_cell_4_77212
identity??#lstm_cell_4/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????(2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_77208lstm_cell_4_77210lstm_cell_4_77212*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_768942%
#lstm_cell_4/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_77208lstm_cell_4_77210lstm_cell_4_77212*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_77221*
condR
while_cond_77220*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_4/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????(:::2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????(
 
_user_specified_nameinputs
?W
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_82173

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_82088*
condR
while_cond_82087*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????2:::2
whilewhile:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?V
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_80708

inputs.
*lstm_cell_3_matmul_readvariableop_resource0
,lstm_cell_3_matmul_1_readvariableop_resource/
+lstm_cell_3_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????(2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????(2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!lstm_cell_3/MatMul/ReadVariableOp?
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul?
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	(?*
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp?
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_1?
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add?
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAddh
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/Const|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dim?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
lstm_cell_3/split?
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid?
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid_1?
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mulz
lstm_cell_3/ReluRelulstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Relu?
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mul_1?
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/add_1?
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid_2y
lstm_cell_3/Relu_1Relulstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Relu_1?
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0 lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????(:?????????(: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_80623*
condR
while_cond_80622*K
output_shapes:
8: : : : :?????????(:?????????(: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????(*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????(2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_79716

inputs5
1lstm_3_lstm_cell_3_matmul_readvariableop_resource7
3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource6
2lstm_3_lstm_cell_3_biasadd_readvariableop_resource5
1lstm_4_lstm_cell_4_matmul_readvariableop_resource7
3lstm_4_lstm_cell_4_matmul_1_readvariableop_resource6
2lstm_4_lstm_cell_4_biasadd_readvariableop_resource5
1lstm_5_lstm_cell_5_matmul_readvariableop_resource7
3lstm_5_lstm_cell_5_matmul_1_readvariableop_resource6
2lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??lstm_3/while?lstm_4/while?lstm_5/whileR
lstm_3/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_3/Shape?
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stack?
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1?
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2?
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicej
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
lstm_3/zeros/mul/y?
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros/Less/y?
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessp
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
lstm_3/zeros/packed/1?
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/Const?
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:?????????(2
lstm_3/zerosn
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
lstm_3/zeros_1/mul/y?
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros_1/Less/y?
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lesst
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
lstm_3/zeros_1/packed/1?
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/Const?
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????(2
lstm_3/zeros_1?
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose/perm?
lstm_3/transpose	Transposeinputslstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
lstm_3/transposed
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:2
lstm_3/Shape_1?
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_1/stack?
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_1?
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_2?
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slice_1?
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_3/TensorArrayV2/element_shape?
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2?
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_3/TensorArrayUnstack/TensorListFromTensor?
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_2/stack?
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_1?
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_2?
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_3/strided_slice_2?
(lstm_3/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp1lstm_3_lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(lstm_3/lstm_cell_3/MatMul/ReadVariableOp?
lstm_3/lstm_cell_3/MatMulMatMullstm_3/strided_slice_2:output:00lstm_3/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul?
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	(?*
dtype02,
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp?
lstm_3/lstm_cell_3/MatMul_1MatMullstm_3/zeros:output:02lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_1?
lstm_3/lstm_cell_3/addAddV2#lstm_3/lstm_cell_3/MatMul:product:0%lstm_3/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/add?
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp?
lstm_3/lstm_cell_3/BiasAddBiasAddlstm_3/lstm_cell_3/add:z:01lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/BiasAddv
lstm_3/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/lstm_cell_3/Const?
"lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_3/lstm_cell_3/split/split_dim?
lstm_3/lstm_cell_3/splitSplit+lstm_3/lstm_cell_3/split/split_dim:output:0#lstm_3/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
lstm_3/lstm_cell_3/split?
lstm_3/lstm_cell_3/SigmoidSigmoid!lstm_3/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/Sigmoid?
lstm_3/lstm_cell_3/Sigmoid_1Sigmoid!lstm_3/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/Sigmoid_1?
lstm_3/lstm_cell_3/mulMul lstm_3/lstm_cell_3/Sigmoid_1:y:0lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/mul?
lstm_3/lstm_cell_3/ReluRelu!lstm_3/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/Relu?
lstm_3/lstm_cell_3/mul_1Mullstm_3/lstm_cell_3/Sigmoid:y:0%lstm_3/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/mul_1?
lstm_3/lstm_cell_3/add_1AddV2lstm_3/lstm_cell_3/mul:z:0lstm_3/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/add_1?
lstm_3/lstm_cell_3/Sigmoid_2Sigmoid!lstm_3/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/Sigmoid_2?
lstm_3/lstm_cell_3/Relu_1Relulstm_3/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/Relu_1?
lstm_3/lstm_cell_3/mul_2Mul lstm_3/lstm_cell_3/Sigmoid_2:y:0'lstm_3/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/mul_2?
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2&
$lstm_3/TensorArrayV2_1/element_shape?
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2_1\
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/time?
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_3/while/maximum_iterationsx
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/while/loop_counter?
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_3_lstm_cell_3_matmul_readvariableop_resource3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource2lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????(:?????????(: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_3_while_body_79327*#
condR
lstm_3_while_cond_79326*K
output_shapes:
8: : : : :?????????(:?????????(: : : : : *
parallel_iterations 2
lstm_3/while?
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   29
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????(*
element_dtype02+
)lstm_3/TensorArrayV2Stack/TensorListStack?
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_3/strided_slice_3/stack?
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_3/strided_slice_3/stack_1?
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_3/stack_2?
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
lstm_3/strided_slice_3?
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose_1/perm?
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????(2
lstm_3/transpose_1t
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/runtimeb
lstm_4/ShapeShapelstm_3/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_4/Shape?
lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_4/strided_slice/stack?
lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_4/strided_slice/stack_1?
lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_4/strided_slice/stack_2?
lstm_4/strided_sliceStridedSlicelstm_4/Shape:output:0#lstm_4/strided_slice/stack:output:0%lstm_4/strided_slice/stack_1:output:0%lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_4/strided_slicej
lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_4/zeros/mul/y?
lstm_4/zeros/mulMullstm_4/strided_slice:output:0lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros/mulm
lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros/Less/y?
lstm_4/zeros/LessLesslstm_4/zeros/mul:z:0lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros/Lessp
lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_4/zeros/packed/1?
lstm_4/zeros/packedPacklstm_4/strided_slice:output:0lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_4/zeros/packedm
lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/zeros/Const?
lstm_4/zerosFilllstm_4/zeros/packed:output:0lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_4/zerosn
lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_4/zeros_1/mul/y?
lstm_4/zeros_1/mulMullstm_4/strided_slice:output:0lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros_1/mulq
lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros_1/Less/y?
lstm_4/zeros_1/LessLesslstm_4/zeros_1/mul:z:0lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros_1/Lesst
lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_4/zeros_1/packed/1?
lstm_4/zeros_1/packedPacklstm_4/strided_slice:output:0 lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_4/zeros_1/packedq
lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/zeros_1/Const?
lstm_4/zeros_1Filllstm_4/zeros_1/packed:output:0lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_4/zeros_1?
lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_4/transpose/perm?
lstm_4/transpose	Transposelstm_3/transpose_1:y:0lstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:?????????(2
lstm_4/transposed
lstm_4/Shape_1Shapelstm_4/transpose:y:0*
T0*
_output_shapes
:2
lstm_4/Shape_1?
lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_4/strided_slice_1/stack?
lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_1/stack_1?
lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_1/stack_2?
lstm_4/strided_slice_1StridedSlicelstm_4/Shape_1:output:0%lstm_4/strided_slice_1/stack:output:0'lstm_4/strided_slice_1/stack_1:output:0'lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_4/strided_slice_1?
"lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_4/TensorArrayV2/element_shape?
lstm_4/TensorArrayV2TensorListReserve+lstm_4/TensorArrayV2/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_4/TensorArrayV2?
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2>
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_4/transpose:y:0Elstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_4/TensorArrayUnstack/TensorListFromTensor?
lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_4/strided_slice_2/stack?
lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_2/stack_1?
lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_2/stack_2?
lstm_4/strided_slice_2StridedSlicelstm_4/transpose:y:0%lstm_4/strided_slice_2/stack:output:0'lstm_4/strided_slice_2/stack_1:output:0'lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
lstm_4/strided_slice_2?
(lstm_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp1lstm_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02*
(lstm_4/lstm_cell_4/MatMul/ReadVariableOp?
lstm_4/lstm_cell_4/MatMulMatMullstm_4/strided_slice_2:output:00lstm_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_4/lstm_cell_4/MatMul?
*lstm_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp3lstm_4_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02,
*lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp?
lstm_4/lstm_cell_4/MatMul_1MatMullstm_4/zeros:output:02lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_4/lstm_cell_4/MatMul_1?
lstm_4/lstm_cell_4/addAddV2#lstm_4/lstm_cell_4/MatMul:product:0%lstm_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_4/lstm_cell_4/add?
)lstm_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp2lstm_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp?
lstm_4/lstm_cell_4/BiasAddBiasAddlstm_4/lstm_cell_4/add:z:01lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_4/lstm_cell_4/BiasAddv
lstm_4/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_4/lstm_cell_4/Const?
"lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_4/lstm_cell_4/split/split_dim?
lstm_4/lstm_cell_4/splitSplit+lstm_4/lstm_cell_4/split/split_dim:output:0#lstm_4/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_4/lstm_cell_4/split?
lstm_4/lstm_cell_4/SigmoidSigmoid!lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/Sigmoid?
lstm_4/lstm_cell_4/Sigmoid_1Sigmoid!lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/Sigmoid_1?
lstm_4/lstm_cell_4/mulMul lstm_4/lstm_cell_4/Sigmoid_1:y:0lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/mul?
lstm_4/lstm_cell_4/ReluRelu!lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/Relu?
lstm_4/lstm_cell_4/mul_1Mullstm_4/lstm_cell_4/Sigmoid:y:0%lstm_4/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/mul_1?
lstm_4/lstm_cell_4/add_1AddV2lstm_4/lstm_cell_4/mul:z:0lstm_4/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/add_1?
lstm_4/lstm_cell_4/Sigmoid_2Sigmoid!lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/Sigmoid_2?
lstm_4/lstm_cell_4/Relu_1Relulstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/Relu_1?
lstm_4/lstm_cell_4/mul_2Mul lstm_4/lstm_cell_4/Sigmoid_2:y:0'lstm_4/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/mul_2?
$lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2&
$lstm_4/TensorArrayV2_1/element_shape?
lstm_4/TensorArrayV2_1TensorListReserve-lstm_4/TensorArrayV2_1/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_4/TensorArrayV2_1\
lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/time?
lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_4/while/maximum_iterationsx
lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/while/loop_counter?
lstm_4/whileWhile"lstm_4/while/loop_counter:output:0(lstm_4/while/maximum_iterations:output:0lstm_4/time:output:0lstm_4/TensorArrayV2_1:handle:0lstm_4/zeros:output:0lstm_4/zeros_1:output:0lstm_4/strided_slice_1:output:0>lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_4_lstm_cell_4_matmul_readvariableop_resource3lstm_4_lstm_cell_4_matmul_1_readvariableop_resource2lstm_4_lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_4_while_body_79476*#
condR
lstm_4_while_cond_79475*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
lstm_4/while?
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_4/TensorArrayV2Stack/TensorListStackTensorListStacklstm_4/while:output:3@lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02+
)lstm_4/TensorArrayV2Stack/TensorListStack?
lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_4/strided_slice_3/stack?
lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_4/strided_slice_3/stack_1?
lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_3/stack_2?
lstm_4/strided_slice_3StridedSlice2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_4/strided_slice_3/stack:output:0'lstm_4/strided_slice_3/stack_1:output:0'lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_4/strided_slice_3?
lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_4/transpose_1/perm?
lstm_4/transpose_1	Transpose2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
lstm_4/transpose_1t
lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/runtimeb
lstm_5/ShapeShapelstm_4/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_5/Shape?
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice/stack?
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_1?
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_2?
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slicej
lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros/mul/y?
lstm_5/zeros/mulMullstm_5/strided_slice:output:0lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/mulm
lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros/Less/y?
lstm_5/zeros/LessLesslstm_5/zeros/mul:z:0lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/Lessp
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros/packed/1?
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros/packedm
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros/Const?
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/zerosn
lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros_1/mul/y?
lstm_5/zeros_1/mulMullstm_5/strided_slice:output:0lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/mulq
lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros_1/Less/y?
lstm_5/zeros_1/LessLesslstm_5/zeros_1/mul:z:0lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/Lesst
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros_1/packed/1?
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros_1/packedq
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros_1/Const?
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/zeros_1?
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose/perm?
lstm_5/transpose	Transposelstm_4/transpose_1:y:0lstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????22
lstm_5/transposed
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:2
lstm_5/Shape_1?
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_1/stack?
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_1?
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_2?
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slice_1?
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_5/TensorArrayV2/element_shape?
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2?
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2>
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_5/TensorArrayUnstack/TensorListFromTensor?
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_2/stack?
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_1?
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_2?
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_5/strided_slice_2?
(lstm_5/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02*
(lstm_5/lstm_cell_5/MatMul/ReadVariableOp?
lstm_5/lstm_cell_5/MatMulMatMullstm_5/strided_slice_2:output:00lstm_5/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/MatMul?
*lstm_5/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02,
*lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_5/lstm_cell_5/MatMul_1MatMullstm_5/zeros:output:02lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/MatMul_1?
lstm_5/lstm_cell_5/addAddV2#lstm_5/lstm_cell_5/MatMul:product:0%lstm_5/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/add?
)lstm_5/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp2lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_5/lstm_cell_5/BiasAddBiasAddlstm_5/lstm_cell_5/add:z:01lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/BiasAddv
lstm_5/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/lstm_cell_5/Const?
"lstm_5/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_5/lstm_cell_5/split/split_dim?
lstm_5/lstm_cell_5/splitSplit+lstm_5/lstm_cell_5/split/split_dim:output:0#lstm_5/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_5/lstm_cell_5/split?
lstm_5/lstm_cell_5/SigmoidSigmoid!lstm_5/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/Sigmoid?
lstm_5/lstm_cell_5/Sigmoid_1Sigmoid!lstm_5/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/Sigmoid_1?
lstm_5/lstm_cell_5/mulMul lstm_5/lstm_cell_5/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/mul?
lstm_5/lstm_cell_5/ReluRelu!lstm_5/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/Relu?
lstm_5/lstm_cell_5/mul_1Mullstm_5/lstm_cell_5/Sigmoid:y:0%lstm_5/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/mul_1?
lstm_5/lstm_cell_5/add_1AddV2lstm_5/lstm_cell_5/mul:z:0lstm_5/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/add_1?
lstm_5/lstm_cell_5/Sigmoid_2Sigmoid!lstm_5/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/Sigmoid_2?
lstm_5/lstm_cell_5/Relu_1Relulstm_5/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/Relu_1?
lstm_5/lstm_cell_5/mul_2Mul lstm_5/lstm_cell_5/Sigmoid_2:y:0'lstm_5/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/mul_2?
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2&
$lstm_5/TensorArrayV2_1/element_shape?
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2_1\
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/time?
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_5/while/maximum_iterationsx
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/while/loop_counter?
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_5_lstm_cell_5_matmul_readvariableop_resource3lstm_5_lstm_cell_5_matmul_1_readvariableop_resource2lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_5_while_body_79625*#
condR
lstm_5_while_cond_79624*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
lstm_5/while?
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02+
)lstm_5/TensorArrayV2Stack/TensorListStack?
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_5/strided_slice_3/stack?
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_5/strided_slice_3/stack_1?
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_3/stack_2?
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_5/strided_slice_3?
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose_1/perm?
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
lstm_5/transpose_1t
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/runtime?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMullstm_5/strided_slice_3:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_1/BiasAdd?
IdentityIdentitydense_1/BiasAdd:output:0^lstm_3/while^lstm_4/while^lstm_5/while*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::2
lstm_3/whilelstm_3/while2
lstm_4/whilelstm_4/while2
lstm_5/whilelstm_5/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_79140

inputs
lstm_3_79113
lstm_3_79115
lstm_3_79117
lstm_4_79120
lstm_4_79122
lstm_4_79124
lstm_5_79127
lstm_5_79129
lstm_5_79131
dense_1_79134
dense_1_79136
identity??dense_1/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?lstm_4/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCallinputslstm_3_79113lstm_3_79115lstm_3_79117*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????(*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_781972 
lstm_3/StatefulPartitionedCall?
lstm_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0lstm_4_79120lstm_4_79122lstm_4_79124*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_785322 
lstm_4/StatefulPartitionedCall?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0lstm_5_79127lstm_5_79129lstm_5_79131*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_788672 
lstm_5/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_1_79134dense_1_79136*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_790602!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_lstm_cell_4_layer_call_fn_82397

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_768942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????(:?????????2:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?
?
while_cond_82087
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_82087___redundant_placeholder03
/while_while_cond_82087___redundant_placeholder13
/while_while_cond_82087___redundant_placeholder23
/while_while_cond_82087___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
+__inference_lstm_cell_5_layer_call_fn_82497

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_775042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????2:?????????2:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?
?
while_cond_81103
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_81103___redundant_placeholder03
/while_while_cond_81103___redundant_placeholder13
/while_while_cond_81103___redundant_placeholder23
/while_while_cond_81103___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
$sequential_1_lstm_5_while_cond_76119D
@sequential_1_lstm_5_while_sequential_1_lstm_5_while_loop_counterJ
Fsequential_1_lstm_5_while_sequential_1_lstm_5_while_maximum_iterations)
%sequential_1_lstm_5_while_placeholder+
'sequential_1_lstm_5_while_placeholder_1+
'sequential_1_lstm_5_while_placeholder_2+
'sequential_1_lstm_5_while_placeholder_3F
Bsequential_1_lstm_5_while_less_sequential_1_lstm_5_strided_slice_1[
Wsequential_1_lstm_5_while_sequential_1_lstm_5_while_cond_76119___redundant_placeholder0[
Wsequential_1_lstm_5_while_sequential_1_lstm_5_while_cond_76119___redundant_placeholder1[
Wsequential_1_lstm_5_while_sequential_1_lstm_5_while_cond_76119___redundant_placeholder2[
Wsequential_1_lstm_5_while_sequential_1_lstm_5_while_cond_76119___redundant_placeholder3&
"sequential_1_lstm_5_while_identity
?
sequential_1/lstm_5/while/LessLess%sequential_1_lstm_5_while_placeholderBsequential_1_lstm_5_while_less_sequential_1_lstm_5_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_1/lstm_5/while/Less?
"sequential_1/lstm_5/while/IdentityIdentity"sequential_1/lstm_5/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_1/lstm_5/while/Identity"Q
"sequential_1_lstm_5_while_identity+sequential_1/lstm_5/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?V
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_78350

inputs.
*lstm_cell_3_matmul_readvariableop_resource0
,lstm_cell_3_matmul_1_readvariableop_resource/
+lstm_cell_3_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????(2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????(2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!lstm_cell_3/MatMul/ReadVariableOp?
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul?
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	(?*
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp?
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_1?
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add?
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAddh
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/Const|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dim?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
lstm_cell_3/split?
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid?
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid_1?
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mulz
lstm_cell_3/ReluRelulstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Relu?
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mul_1?
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/add_1?
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid_2y
lstm_cell_3/Relu_1Relulstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Relu_1?
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0 lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????(:?????????(: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_78265*
condR
while_cond_78264*K
output_shapes:
8: : : : :?????????(:?????????(: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????(*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????(2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_lstm_5_layer_call_fn_82184

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_788672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?9
?
while_body_80776
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_3_matmul_readvariableop_resource_08
4while_lstm_cell_3_matmul_1_readvariableop_resource_07
3while_lstm_cell_3_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_3_matmul_readvariableop_resource6
2while_lstm_cell_3_matmul_1_readvariableop_resource5
1while_lstm_cell_3_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOp?
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul?
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	(?*
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOp?
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_1?
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add?
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOp?
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAddt
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_3/Const?
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
while/lstm_cell_3/split?
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid?
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid_1?
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul?
while/lstm_cell_3/ReluRelu while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Relu?
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0$while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul_1?
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/add_1?
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid_2?
while/lstm_cell_3/Relu_1Reluwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Relu_1?
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0&while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0*
T0*'
_output_shapes
:?????????(2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????(:?????????(: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
: 
?9
?
while_body_81760
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_77830
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_77830___redundant_placeholder03
/while_while_cond_77830___redundant_placeholder13
/while_while_cond_77830___redundant_placeholder23
/while_while_cond_77830___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_77220
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_77220___redundant_placeholder03
/while_while_cond_77220___redundant_placeholder13
/while_while_cond_77220___redundant_placeholder23
/while_while_cond_77220___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
$sequential_1_lstm_4_while_cond_75970D
@sequential_1_lstm_4_while_sequential_1_lstm_4_while_loop_counterJ
Fsequential_1_lstm_4_while_sequential_1_lstm_4_while_maximum_iterations)
%sequential_1_lstm_4_while_placeholder+
'sequential_1_lstm_4_while_placeholder_1+
'sequential_1_lstm_4_while_placeholder_2+
'sequential_1_lstm_4_while_placeholder_3F
Bsequential_1_lstm_4_while_less_sequential_1_lstm_4_strided_slice_1[
Wsequential_1_lstm_4_while_sequential_1_lstm_4_while_cond_75970___redundant_placeholder0[
Wsequential_1_lstm_4_while_sequential_1_lstm_4_while_cond_75970___redundant_placeholder1[
Wsequential_1_lstm_4_while_sequential_1_lstm_4_while_cond_75970___redundant_placeholder2[
Wsequential_1_lstm_4_while_sequential_1_lstm_4_while_cond_75970___redundant_placeholder3&
"sequential_1_lstm_4_while_identity
?
sequential_1/lstm_4/while/LessLess%sequential_1_lstm_4_while_placeholderBsequential_1_lstm_4_while_less_sequential_1_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_1/lstm_4/while/Less?
"sequential_1/lstm_4/while/IdentityIdentity"sequential_1/lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_1/lstm_4/while/Identity"Q
"sequential_1_lstm_4_while_identity+sequential_1/lstm_4/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_80447
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_80447___redundant_placeholder03
/while_while_cond_80447___redundant_placeholder13
/while_while_cond_80447___redundant_placeholder23
/while_while_cond_80447___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????(:?????????(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
:
?
?
&__inference_lstm_3_layer_call_fn_80555
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????(*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_768122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????(2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
??
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_80173

inputs5
1lstm_3_lstm_cell_3_matmul_readvariableop_resource7
3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource6
2lstm_3_lstm_cell_3_biasadd_readvariableop_resource5
1lstm_4_lstm_cell_4_matmul_readvariableop_resource7
3lstm_4_lstm_cell_4_matmul_1_readvariableop_resource6
2lstm_4_lstm_cell_4_biasadd_readvariableop_resource5
1lstm_5_lstm_cell_5_matmul_readvariableop_resource7
3lstm_5_lstm_cell_5_matmul_1_readvariableop_resource6
2lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??lstm_3/while?lstm_4/while?lstm_5/whileR
lstm_3/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_3/Shape?
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stack?
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1?
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2?
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicej
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
lstm_3/zeros/mul/y?
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros/Less/y?
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessp
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
lstm_3/zeros/packed/1?
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/Const?
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:?????????(2
lstm_3/zerosn
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
lstm_3/zeros_1/mul/y?
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros_1/Less/y?
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lesst
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
lstm_3/zeros_1/packed/1?
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/Const?
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????(2
lstm_3/zeros_1?
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose/perm?
lstm_3/transpose	Transposeinputslstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
lstm_3/transposed
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:2
lstm_3/Shape_1?
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_1/stack?
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_1?
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_2?
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slice_1?
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_3/TensorArrayV2/element_shape?
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2?
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_3/TensorArrayUnstack/TensorListFromTensor?
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_2/stack?
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_1?
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_2?
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
lstm_3/strided_slice_2?
(lstm_3/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp1lstm_3_lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(lstm_3/lstm_cell_3/MatMul/ReadVariableOp?
lstm_3/lstm_cell_3/MatMulMatMullstm_3/strided_slice_2:output:00lstm_3/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul?
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	(?*
dtype02,
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp?
lstm_3/lstm_cell_3/MatMul_1MatMullstm_3/zeros:output:02lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/MatMul_1?
lstm_3/lstm_cell_3/addAddV2#lstm_3/lstm_cell_3/MatMul:product:0%lstm_3/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/add?
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp?
lstm_3/lstm_cell_3/BiasAddBiasAddlstm_3/lstm_cell_3/add:z:01lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_3/lstm_cell_3/BiasAddv
lstm_3/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/lstm_cell_3/Const?
"lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_3/lstm_cell_3/split/split_dim?
lstm_3/lstm_cell_3/splitSplit+lstm_3/lstm_cell_3/split/split_dim:output:0#lstm_3/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
lstm_3/lstm_cell_3/split?
lstm_3/lstm_cell_3/SigmoidSigmoid!lstm_3/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/Sigmoid?
lstm_3/lstm_cell_3/Sigmoid_1Sigmoid!lstm_3/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/Sigmoid_1?
lstm_3/lstm_cell_3/mulMul lstm_3/lstm_cell_3/Sigmoid_1:y:0lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/mul?
lstm_3/lstm_cell_3/ReluRelu!lstm_3/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/Relu?
lstm_3/lstm_cell_3/mul_1Mullstm_3/lstm_cell_3/Sigmoid:y:0%lstm_3/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/mul_1?
lstm_3/lstm_cell_3/add_1AddV2lstm_3/lstm_cell_3/mul:z:0lstm_3/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/add_1?
lstm_3/lstm_cell_3/Sigmoid_2Sigmoid!lstm_3/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/Sigmoid_2?
lstm_3/lstm_cell_3/Relu_1Relulstm_3/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/Relu_1?
lstm_3/lstm_cell_3/mul_2Mul lstm_3/lstm_cell_3/Sigmoid_2:y:0'lstm_3/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_3/lstm_cell_3/mul_2?
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2&
$lstm_3/TensorArrayV2_1/element_shape?
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2_1\
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/time?
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_3/while/maximum_iterationsx
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/while/loop_counter?
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_3_lstm_cell_3_matmul_readvariableop_resource3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource2lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????(:?????????(: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_3_while_body_79784*#
condR
lstm_3_while_cond_79783*K
output_shapes:
8: : : : :?????????(:?????????(: : : : : *
parallel_iterations 2
lstm_3/while?
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   29
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????(*
element_dtype02+
)lstm_3/TensorArrayV2Stack/TensorListStack?
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_3/strided_slice_3/stack?
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_3/strided_slice_3/stack_1?
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_3/stack_2?
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
lstm_3/strided_slice_3?
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose_1/perm?
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????(2
lstm_3/transpose_1t
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/runtimeb
lstm_4/ShapeShapelstm_3/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_4/Shape?
lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_4/strided_slice/stack?
lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_4/strided_slice/stack_1?
lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_4/strided_slice/stack_2?
lstm_4/strided_sliceStridedSlicelstm_4/Shape:output:0#lstm_4/strided_slice/stack:output:0%lstm_4/strided_slice/stack_1:output:0%lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_4/strided_slicej
lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_4/zeros/mul/y?
lstm_4/zeros/mulMullstm_4/strided_slice:output:0lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros/mulm
lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros/Less/y?
lstm_4/zeros/LessLesslstm_4/zeros/mul:z:0lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros/Lessp
lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_4/zeros/packed/1?
lstm_4/zeros/packedPacklstm_4/strided_slice:output:0lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_4/zeros/packedm
lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/zeros/Const?
lstm_4/zerosFilllstm_4/zeros/packed:output:0lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_4/zerosn
lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_4/zeros_1/mul/y?
lstm_4/zeros_1/mulMullstm_4/strided_slice:output:0lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros_1/mulq
lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_4/zeros_1/Less/y?
lstm_4/zeros_1/LessLesslstm_4/zeros_1/mul:z:0lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_4/zeros_1/Lesst
lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_4/zeros_1/packed/1?
lstm_4/zeros_1/packedPacklstm_4/strided_slice:output:0 lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_4/zeros_1/packedq
lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/zeros_1/Const?
lstm_4/zeros_1Filllstm_4/zeros_1/packed:output:0lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_4/zeros_1?
lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_4/transpose/perm?
lstm_4/transpose	Transposelstm_3/transpose_1:y:0lstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:?????????(2
lstm_4/transposed
lstm_4/Shape_1Shapelstm_4/transpose:y:0*
T0*
_output_shapes
:2
lstm_4/Shape_1?
lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_4/strided_slice_1/stack?
lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_1/stack_1?
lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_1/stack_2?
lstm_4/strided_slice_1StridedSlicelstm_4/Shape_1:output:0%lstm_4/strided_slice_1/stack:output:0'lstm_4/strided_slice_1/stack_1:output:0'lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_4/strided_slice_1?
"lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_4/TensorArrayV2/element_shape?
lstm_4/TensorArrayV2TensorListReserve+lstm_4/TensorArrayV2/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_4/TensorArrayV2?
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2>
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_4/transpose:y:0Elstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_4/TensorArrayUnstack/TensorListFromTensor?
lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_4/strided_slice_2/stack?
lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_2/stack_1?
lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_2/stack_2?
lstm_4/strided_slice_2StridedSlicelstm_4/transpose:y:0%lstm_4/strided_slice_2/stack:output:0'lstm_4/strided_slice_2/stack_1:output:0'lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
lstm_4/strided_slice_2?
(lstm_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp1lstm_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02*
(lstm_4/lstm_cell_4/MatMul/ReadVariableOp?
lstm_4/lstm_cell_4/MatMulMatMullstm_4/strided_slice_2:output:00lstm_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_4/lstm_cell_4/MatMul?
*lstm_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp3lstm_4_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02,
*lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp?
lstm_4/lstm_cell_4/MatMul_1MatMullstm_4/zeros:output:02lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_4/lstm_cell_4/MatMul_1?
lstm_4/lstm_cell_4/addAddV2#lstm_4/lstm_cell_4/MatMul:product:0%lstm_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_4/lstm_cell_4/add?
)lstm_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp2lstm_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp?
lstm_4/lstm_cell_4/BiasAddBiasAddlstm_4/lstm_cell_4/add:z:01lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_4/lstm_cell_4/BiasAddv
lstm_4/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_4/lstm_cell_4/Const?
"lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_4/lstm_cell_4/split/split_dim?
lstm_4/lstm_cell_4/splitSplit+lstm_4/lstm_cell_4/split/split_dim:output:0#lstm_4/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_4/lstm_cell_4/split?
lstm_4/lstm_cell_4/SigmoidSigmoid!lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/Sigmoid?
lstm_4/lstm_cell_4/Sigmoid_1Sigmoid!lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/Sigmoid_1?
lstm_4/lstm_cell_4/mulMul lstm_4/lstm_cell_4/Sigmoid_1:y:0lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/mul?
lstm_4/lstm_cell_4/ReluRelu!lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/Relu?
lstm_4/lstm_cell_4/mul_1Mullstm_4/lstm_cell_4/Sigmoid:y:0%lstm_4/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/mul_1?
lstm_4/lstm_cell_4/add_1AddV2lstm_4/lstm_cell_4/mul:z:0lstm_4/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/add_1?
lstm_4/lstm_cell_4/Sigmoid_2Sigmoid!lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/Sigmoid_2?
lstm_4/lstm_cell_4/Relu_1Relulstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/Relu_1?
lstm_4/lstm_cell_4/mul_2Mul lstm_4/lstm_cell_4/Sigmoid_2:y:0'lstm_4/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_4/lstm_cell_4/mul_2?
$lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2&
$lstm_4/TensorArrayV2_1/element_shape?
lstm_4/TensorArrayV2_1TensorListReserve-lstm_4/TensorArrayV2_1/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_4/TensorArrayV2_1\
lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/time?
lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_4/while/maximum_iterationsx
lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_4/while/loop_counter?
lstm_4/whileWhile"lstm_4/while/loop_counter:output:0(lstm_4/while/maximum_iterations:output:0lstm_4/time:output:0lstm_4/TensorArrayV2_1:handle:0lstm_4/zeros:output:0lstm_4/zeros_1:output:0lstm_4/strided_slice_1:output:0>lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_4_lstm_cell_4_matmul_readvariableop_resource3lstm_4_lstm_cell_4_matmul_1_readvariableop_resource2lstm_4_lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_4_while_body_79933*#
condR
lstm_4_while_cond_79932*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
lstm_4/while?
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_4/TensorArrayV2Stack/TensorListStackTensorListStacklstm_4/while:output:3@lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02+
)lstm_4/TensorArrayV2Stack/TensorListStack?
lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_4/strided_slice_3/stack?
lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_4/strided_slice_3/stack_1?
lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_4/strided_slice_3/stack_2?
lstm_4/strided_slice_3StridedSlice2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_4/strided_slice_3/stack:output:0'lstm_4/strided_slice_3/stack_1:output:0'lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_4/strided_slice_3?
lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_4/transpose_1/perm?
lstm_4/transpose_1	Transpose2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
lstm_4/transpose_1t
lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/runtimeb
lstm_5/ShapeShapelstm_4/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_5/Shape?
lstm_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice/stack?
lstm_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_1?
lstm_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_5/strided_slice/stack_2?
lstm_5/strided_sliceStridedSlicelstm_5/Shape:output:0#lstm_5/strided_slice/stack:output:0%lstm_5/strided_slice/stack_1:output:0%lstm_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slicej
lstm_5/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros/mul/y?
lstm_5/zeros/mulMullstm_5/strided_slice:output:0lstm_5/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/mulm
lstm_5/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros/Less/y?
lstm_5/zeros/LessLesslstm_5/zeros/mul:z:0lstm_5/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros/Lessp
lstm_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros/packed/1?
lstm_5/zeros/packedPacklstm_5/strided_slice:output:0lstm_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros/packedm
lstm_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros/Const?
lstm_5/zerosFilllstm_5/zeros/packed:output:0lstm_5/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/zerosn
lstm_5/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros_1/mul/y?
lstm_5/zeros_1/mulMullstm_5/strided_slice:output:0lstm_5/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/mulq
lstm_5/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_5/zeros_1/Less/y?
lstm_5/zeros_1/LessLesslstm_5/zeros_1/mul:z:0lstm_5/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_5/zeros_1/Lesst
lstm_5/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
lstm_5/zeros_1/packed/1?
lstm_5/zeros_1/packedPacklstm_5/strided_slice:output:0 lstm_5/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_5/zeros_1/packedq
lstm_5/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/zeros_1/Const?
lstm_5/zeros_1Filllstm_5/zeros_1/packed:output:0lstm_5/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/zeros_1?
lstm_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose/perm?
lstm_5/transpose	Transposelstm_4/transpose_1:y:0lstm_5/transpose/perm:output:0*
T0*+
_output_shapes
:?????????22
lstm_5/transposed
lstm_5/Shape_1Shapelstm_5/transpose:y:0*
T0*
_output_shapes
:2
lstm_5/Shape_1?
lstm_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_1/stack?
lstm_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_1?
lstm_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_1/stack_2?
lstm_5/strided_slice_1StridedSlicelstm_5/Shape_1:output:0%lstm_5/strided_slice_1/stack:output:0'lstm_5/strided_slice_1/stack_1:output:0'lstm_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_5/strided_slice_1?
"lstm_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"lstm_5/TensorArrayV2/element_shape?
lstm_5/TensorArrayV2TensorListReserve+lstm_5/TensorArrayV2/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2?
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2>
<lstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape?
.lstm_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_5/transpose:y:0Elstm_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_5/TensorArrayUnstack/TensorListFromTensor?
lstm_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_5/strided_slice_2/stack?
lstm_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_1?
lstm_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_2/stack_2?
lstm_5/strided_slice_2StridedSlicelstm_5/transpose:y:0%lstm_5/strided_slice_2/stack:output:0'lstm_5/strided_slice_2/stack_1:output:0'lstm_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_5/strided_slice_2?
(lstm_5/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp1lstm_5_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02*
(lstm_5/lstm_cell_5/MatMul/ReadVariableOp?
lstm_5/lstm_cell_5/MatMulMatMullstm_5/strided_slice_2:output:00lstm_5/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/MatMul?
*lstm_5/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp3lstm_5_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02,
*lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_5/lstm_cell_5/MatMul_1MatMullstm_5/zeros:output:02lstm_5/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/MatMul_1?
lstm_5/lstm_cell_5/addAddV2#lstm_5/lstm_cell_5/MatMul:product:0%lstm_5/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/add?
)lstm_5/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp2lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_5/lstm_cell_5/BiasAddBiasAddlstm_5/lstm_cell_5/add:z:01lstm_5/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_5/lstm_cell_5/BiasAddv
lstm_5/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_5/lstm_cell_5/Const?
"lstm_5/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_5/lstm_cell_5/split/split_dim?
lstm_5/lstm_cell_5/splitSplit+lstm_5/lstm_cell_5/split/split_dim:output:0#lstm_5/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_5/lstm_cell_5/split?
lstm_5/lstm_cell_5/SigmoidSigmoid!lstm_5/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/Sigmoid?
lstm_5/lstm_cell_5/Sigmoid_1Sigmoid!lstm_5/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/Sigmoid_1?
lstm_5/lstm_cell_5/mulMul lstm_5/lstm_cell_5/Sigmoid_1:y:0lstm_5/zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/mul?
lstm_5/lstm_cell_5/ReluRelu!lstm_5/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/Relu?
lstm_5/lstm_cell_5/mul_1Mullstm_5/lstm_cell_5/Sigmoid:y:0%lstm_5/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/mul_1?
lstm_5/lstm_cell_5/add_1AddV2lstm_5/lstm_cell_5/mul:z:0lstm_5/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/add_1?
lstm_5/lstm_cell_5/Sigmoid_2Sigmoid!lstm_5/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/Sigmoid_2?
lstm_5/lstm_cell_5/Relu_1Relulstm_5/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/Relu_1?
lstm_5/lstm_cell_5/mul_2Mul lstm_5/lstm_cell_5/Sigmoid_2:y:0'lstm_5/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_5/lstm_cell_5/mul_2?
$lstm_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2&
$lstm_5/TensorArrayV2_1/element_shape?
lstm_5/TensorArrayV2_1TensorListReserve-lstm_5/TensorArrayV2_1/element_shape:output:0lstm_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_5/TensorArrayV2_1\
lstm_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/time?
lstm_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
lstm_5/while/maximum_iterationsx
lstm_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_5/while/loop_counter?
lstm_5/whileWhile"lstm_5/while/loop_counter:output:0(lstm_5/while/maximum_iterations:output:0lstm_5/time:output:0lstm_5/TensorArrayV2_1:handle:0lstm_5/zeros:output:0lstm_5/zeros_1:output:0lstm_5/strided_slice_1:output:0>lstm_5/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_5_lstm_cell_5_matmul_readvariableop_resource3lstm_5_lstm_cell_5_matmul_1_readvariableop_resource2lstm_5_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_5_while_body_80082*#
condR
lstm_5_while_cond_80081*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
lstm_5/while?
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7lstm_5/TensorArrayV2Stack/TensorListStack/element_shape?
)lstm_5/TensorArrayV2Stack/TensorListStackTensorListStacklstm_5/while:output:3@lstm_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02+
)lstm_5/TensorArrayV2Stack/TensorListStack?
lstm_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_5/strided_slice_3/stack?
lstm_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_5/strided_slice_3/stack_1?
lstm_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_5/strided_slice_3/stack_2?
lstm_5/strided_slice_3StridedSlice2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_5/strided_slice_3/stack:output:0'lstm_5/strided_slice_3/stack_1:output:0'lstm_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
lstm_5/strided_slice_3?
lstm_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_5/transpose_1/perm?
lstm_5/transpose_1	Transpose2lstm_5/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
lstm_5/transpose_1t
lstm_5/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_5/runtime?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:2(*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMullstm_5/strided_slice_3:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_1/BiasAdd?
IdentityIdentitydense_1/BiasAdd:output:0^lstm_3/while^lstm_4/while^lstm_5/while*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::2
lstm_3/whilelstm_3/while2
lstm_4/whilelstm_4/while2
lstm_5/whilelstm_5/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
lstm_4_while_cond_79475*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3,
(lstm_4_while_less_lstm_4_strided_slice_1A
=lstm_4_while_lstm_4_while_cond_79475___redundant_placeholder0A
=lstm_4_while_lstm_4_while_cond_79475___redundant_placeholder1A
=lstm_4_while_lstm_4_while_cond_79475___redundant_placeholder2A
=lstm_4_while_lstm_4_while_cond_79475___redundant_placeholder3
lstm_4_while_identity
?
lstm_4/while/LessLesslstm_4_while_placeholder(lstm_4_while_less_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2
lstm_4/while/Lessr
lstm_4/while/IdentityIdentitylstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_4/while/Identity"7
lstm_4_while_identitylstm_4/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?9
?
while_body_80448
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_3_matmul_readvariableop_resource_08
4while_lstm_cell_3_matmul_1_readvariableop_resource_07
3while_lstm_cell_3_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_3_matmul_readvariableop_resource6
2while_lstm_cell_3_matmul_1_readvariableop_resource5
1while_lstm_cell_3_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOp?
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul?
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	(?*
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOp?
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_1?
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add?
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOp?
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAddt
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_3/Const?
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
while/lstm_cell_3/split?
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid?
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid_1?
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul?
while/lstm_cell_3/ReluRelu while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Relu?
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0$while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul_1?
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/add_1?
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid_2?
while/lstm_cell_3/Relu_1Reluwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Relu_1?
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0&while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0*
T0*'
_output_shapes
:?????????(2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????(:?????????(: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_76742
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_76742___redundant_placeholder03
/while_while_cond_76742___redundant_placeholder13
/while_while_cond_76742___redundant_placeholder23
/while_while_cond_76742___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????(:?????????(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
:
?
?
&__inference_lstm_4_layer_call_fn_81528

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_785322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????(
 
_user_specified_nameinputs
?$
?
while_body_76743
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_3_76767_0
while_lstm_cell_3_76769_0
while_lstm_cell_3_76771_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_3_76767
while_lstm_cell_3_76769
while_lstm_cell_3_76771??)while/lstm_cell_3/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_76767_0while_lstm_cell_3_76769_0while_lstm_cell_3_76771_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????(:?????????(:?????????(*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_763172+
)while/lstm_cell_3/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1*^while/lstm_cell_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2*^while/lstm_cell_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_3_76767while_lstm_cell_3_76767_0"4
while_lstm_cell_3_76769while_lstm_cell_3_76769_0"4
while_lstm_cell_3_76771while_lstm_cell_3_76771_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????(:?????????(: : :::2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
: 
?	
?
,__inference_sequential_1_layer_call_fn_80200

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_791402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?D
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_77900

inputs
lstm_cell_5_77818
lstm_cell_5_77820
lstm_cell_5_77822
identity??#lstm_cell_5/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5_77818lstm_cell_5_77820lstm_cell_5_77822*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_775042%
#lstm_cell_5/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5_77818lstm_cell_5_77820lstm_cell_5_77822*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_77831*
condR
while_cond_77830*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_5/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::2J
#lstm_cell_5/StatefulPartitionedCall#lstm_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????2
 
_user_specified_nameinputs
?
?
&__inference_lstm_4_layer_call_fn_81200
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_772902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????(:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????(
"
_user_specified_name
inputs/0
?9
?
while_body_82088
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
$sequential_1_lstm_3_while_cond_75821D
@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counterJ
Fsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations)
%sequential_1_lstm_3_while_placeholder+
'sequential_1_lstm_3_while_placeholder_1+
'sequential_1_lstm_3_while_placeholder_2+
'sequential_1_lstm_3_while_placeholder_3F
Bsequential_1_lstm_3_while_less_sequential_1_lstm_3_strided_slice_1[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_75821___redundant_placeholder0[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_75821___redundant_placeholder1[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_75821___redundant_placeholder2[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_75821___redundant_placeholder3&
"sequential_1_lstm_3_while_identity
?
sequential_1/lstm_3/while/LessLess%sequential_1_lstm_3_while_placeholderBsequential_1_lstm_3_while_less_sequential_1_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_1/lstm_3/while/Less?
"sequential_1/lstm_3/while/IdentityIdentity"sequential_1/lstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_1/lstm_3/while/Identity"Q
"sequential_1_lstm_3_while_identity+sequential_1/lstm_3/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????(:?????????(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
:
?R
?
$sequential_1_lstm_5_while_body_76120D
@sequential_1_lstm_5_while_sequential_1_lstm_5_while_loop_counterJ
Fsequential_1_lstm_5_while_sequential_1_lstm_5_while_maximum_iterations)
%sequential_1_lstm_5_while_placeholder+
'sequential_1_lstm_5_while_placeholder_1+
'sequential_1_lstm_5_while_placeholder_2+
'sequential_1_lstm_5_while_placeholder_3C
?sequential_1_lstm_5_while_sequential_1_lstm_5_strided_slice_1_0
{sequential_1_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_5_tensorarrayunstack_tensorlistfromtensor_0J
Fsequential_1_lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0L
Hsequential_1_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0K
Gsequential_1_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0&
"sequential_1_lstm_5_while_identity(
$sequential_1_lstm_5_while_identity_1(
$sequential_1_lstm_5_while_identity_2(
$sequential_1_lstm_5_while_identity_3(
$sequential_1_lstm_5_while_identity_4(
$sequential_1_lstm_5_while_identity_5A
=sequential_1_lstm_5_while_sequential_1_lstm_5_strided_slice_1}
ysequential_1_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_5_tensorarrayunstack_tensorlistfromtensorH
Dsequential_1_lstm_5_while_lstm_cell_5_matmul_readvariableop_resourceJ
Fsequential_1_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resourceI
Esequential_1_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource??
Ksequential_1/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2M
Ksequential_1/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=sequential_1/lstm_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_5_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_5_while_placeholderTsequential_1/lstm_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02?
=sequential_1/lstm_5/while/TensorArrayV2Read/TensorListGetItem?
;sequential_1/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOpFsequential_1_lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02=
;sequential_1/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp?
,sequential_1/lstm_5/while/lstm_cell_5/MatMulMatMulDsequential_1/lstm_5/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_1/lstm_5/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,sequential_1/lstm_5/while/lstm_cell_5/MatMul?
=sequential_1/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOpHsequential_1_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02?
=sequential_1/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp?
.sequential_1/lstm_5/while/lstm_cell_5/MatMul_1MatMul'sequential_1_lstm_5_while_placeholder_2Esequential_1/lstm_5/while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_1/lstm_5/while/lstm_cell_5/MatMul_1?
)sequential_1/lstm_5/while/lstm_cell_5/addAddV26sequential_1/lstm_5/while/lstm_cell_5/MatMul:product:08sequential_1/lstm_5/while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2+
)sequential_1/lstm_5/while/lstm_cell_5/add?
<sequential_1/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOpGsequential_1_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02>
<sequential_1/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp?
-sequential_1/lstm_5/while/lstm_cell_5/BiasAddBiasAdd-sequential_1/lstm_5/while/lstm_cell_5/add:z:0Dsequential_1/lstm_5/while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-sequential_1/lstm_5/while/lstm_cell_5/BiasAdd?
+sequential_1/lstm_5/while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+sequential_1/lstm_5/while/lstm_cell_5/Const?
5sequential_1/lstm_5/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_1/lstm_5/while/lstm_cell_5/split/split_dim?
+sequential_1/lstm_5/while/lstm_cell_5/splitSplit>sequential_1/lstm_5/while/lstm_cell_5/split/split_dim:output:06sequential_1/lstm_5/while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2-
+sequential_1/lstm_5/while/lstm_cell_5/split?
-sequential_1/lstm_5/while/lstm_cell_5/SigmoidSigmoid4sequential_1/lstm_5/while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22/
-sequential_1/lstm_5/while/lstm_cell_5/Sigmoid?
/sequential_1/lstm_5/while/lstm_cell_5/Sigmoid_1Sigmoid4sequential_1/lstm_5/while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????221
/sequential_1/lstm_5/while/lstm_cell_5/Sigmoid_1?
)sequential_1/lstm_5/while/lstm_cell_5/mulMul3sequential_1/lstm_5/while/lstm_cell_5/Sigmoid_1:y:0'sequential_1_lstm_5_while_placeholder_3*
T0*'
_output_shapes
:?????????22+
)sequential_1/lstm_5/while/lstm_cell_5/mul?
*sequential_1/lstm_5/while/lstm_cell_5/ReluRelu4sequential_1/lstm_5/while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22,
*sequential_1/lstm_5/while/lstm_cell_5/Relu?
+sequential_1/lstm_5/while/lstm_cell_5/mul_1Mul1sequential_1/lstm_5/while/lstm_cell_5/Sigmoid:y:08sequential_1/lstm_5/while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22-
+sequential_1/lstm_5/while/lstm_cell_5/mul_1?
+sequential_1/lstm_5/while/lstm_cell_5/add_1AddV2-sequential_1/lstm_5/while/lstm_cell_5/mul:z:0/sequential_1/lstm_5/while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22-
+sequential_1/lstm_5/while/lstm_cell_5/add_1?
/sequential_1/lstm_5/while/lstm_cell_5/Sigmoid_2Sigmoid4sequential_1/lstm_5/while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????221
/sequential_1/lstm_5/while/lstm_cell_5/Sigmoid_2?
,sequential_1/lstm_5/while/lstm_cell_5/Relu_1Relu/sequential_1/lstm_5/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22.
,sequential_1/lstm_5/while/lstm_cell_5/Relu_1?
+sequential_1/lstm_5/while/lstm_cell_5/mul_2Mul3sequential_1/lstm_5/while/lstm_cell_5/Sigmoid_2:y:0:sequential_1/lstm_5/while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22-
+sequential_1/lstm_5/while/lstm_cell_5/mul_2?
>sequential_1/lstm_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_5_while_placeholder_1%sequential_1_lstm_5_while_placeholder/sequential_1/lstm_5/while/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_1/lstm_5/while/TensorArrayV2Write/TensorListSetItem?
sequential_1/lstm_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_1/lstm_5/while/add/y?
sequential_1/lstm_5/while/addAddV2%sequential_1_lstm_5_while_placeholder(sequential_1/lstm_5/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_5/while/add?
!sequential_1/lstm_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_1/lstm_5/while/add_1/y?
sequential_1/lstm_5/while/add_1AddV2@sequential_1_lstm_5_while_sequential_1_lstm_5_while_loop_counter*sequential_1/lstm_5/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_5/while/add_1?
"sequential_1/lstm_5/while/IdentityIdentity#sequential_1/lstm_5/while/add_1:z:0*
T0*
_output_shapes
: 2$
"sequential_1/lstm_5/while/Identity?
$sequential_1/lstm_5/while/Identity_1IdentityFsequential_1_lstm_5_while_sequential_1_lstm_5_while_maximum_iterations*
T0*
_output_shapes
: 2&
$sequential_1/lstm_5/while/Identity_1?
$sequential_1/lstm_5/while/Identity_2Identity!sequential_1/lstm_5/while/add:z:0*
T0*
_output_shapes
: 2&
$sequential_1/lstm_5/while/Identity_2?
$sequential_1/lstm_5/while/Identity_3IdentityNsequential_1/lstm_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2&
$sequential_1/lstm_5/while/Identity_3?
$sequential_1/lstm_5/while/Identity_4Identity/sequential_1/lstm_5/while/lstm_cell_5/mul_2:z:0*
T0*'
_output_shapes
:?????????22&
$sequential_1/lstm_5/while/Identity_4?
$sequential_1/lstm_5/while/Identity_5Identity/sequential_1/lstm_5/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22&
$sequential_1/lstm_5/while/Identity_5"Q
"sequential_1_lstm_5_while_identity+sequential_1/lstm_5/while/Identity:output:0"U
$sequential_1_lstm_5_while_identity_1-sequential_1/lstm_5/while/Identity_1:output:0"U
$sequential_1_lstm_5_while_identity_2-sequential_1/lstm_5/while/Identity_2:output:0"U
$sequential_1_lstm_5_while_identity_3-sequential_1/lstm_5/while/Identity_3:output:0"U
$sequential_1_lstm_5_while_identity_4-sequential_1/lstm_5/while/Identity_4:output:0"U
$sequential_1_lstm_5_while_identity_5-sequential_1/lstm_5/while/Identity_5:output:0"?
Esequential_1_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resourceGsequential_1_lstm_5_while_lstm_cell_5_biasadd_readvariableop_resource_0"?
Fsequential_1_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resourceHsequential_1_lstm_5_while_lstm_cell_5_matmul_1_readvariableop_resource_0"?
Dsequential_1_lstm_5_while_lstm_cell_5_matmul_readvariableop_resourceFsequential_1_lstm_5_while_lstm_cell_5_matmul_readvariableop_resource_0"?
=sequential_1_lstm_5_while_sequential_1_lstm_5_strided_slice_1?sequential_1_lstm_5_while_sequential_1_lstm_5_strided_slice_1_0"?
ysequential_1_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_5_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_5_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_5_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_78934
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_78934___redundant_placeholder03
/while_while_cond_78934___redundant_placeholder13
/while_while_cond_78934___redundant_placeholder23
/while_while_cond_78934___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_82205

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2:::O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_82447

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????2:?????????2:?????????2::::O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?
?
while_cond_81934
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_81934___redundant_placeholder03
/while_while_cond_81934___redundant_placeholder13
/while_while_cond_81934___redundant_placeholder23
/while_while_cond_81934___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?V
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_81517

inputs.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????(2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/MatMul?
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp?
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/MatMul_1?
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/add?
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dim?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_4/split?
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid?
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid_1?
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Relu?
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mul_1?
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/add_1?
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Relu_1?
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_81432*
condR
while_cond_81431*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(:::2
whilewhile:S O
+
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_82347

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????(:?????????2:?????????2::::O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?
?
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_82480

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????22	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????22
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????22
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????22
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????22
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????22
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????22
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
mul_2]
IdentityIdentity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	mul_2:z:0*
T0*'
_output_shapes
:?????????22

Identity_1a

Identity_2Identity	add_1:z:0*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????2:?????????2:?????????2::::O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?
?
&__inference_lstm_4_layer_call_fn_81211
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_774222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????(:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????(
"
_user_specified_name
inputs/0
?D
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_76812

inputs
lstm_cell_3_76730
lstm_cell_3_76732
lstm_cell_3_76734
identity??#lstm_cell_3/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????(2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????(2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_76730lstm_cell_3_76732lstm_cell_3_76734*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????(:?????????(:?????????(*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_763172%
#lstm_cell_3/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_76730lstm_cell_3_76732lstm_cell_3_76734*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????(:?????????(: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_76743*
condR
while_cond_76742*K
output_shapes:
8: : : : :?????????(:?????????(: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????(*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????(2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitytranspose_1:y:0$^lstm_cell_3/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????(2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_79107
lstm_3_input
lstm_3_79080
lstm_3_79082
lstm_3_79084
lstm_4_79087
lstm_4_79089
lstm_4_79091
lstm_5_79094
lstm_5_79096
lstm_5_79098
dense_1_79101
dense_1_79103
identity??dense_1/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?lstm_4/StatefulPartitionedCall?lstm_5/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCalllstm_3_inputlstm_3_79080lstm_3_79082lstm_3_79084*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????(*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_783502 
lstm_3/StatefulPartitionedCall?
lstm_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0lstm_4_79087lstm_4_79089lstm_4_79091*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_4_layer_call_and_return_conditional_losses_786852 
lstm_4/StatefulPartitionedCall?
lstm_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0lstm_5_79094lstm_5_79096lstm_5_79098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_5_layer_call_and_return_conditional_losses_790202 
lstm_5/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall'lstm_5/StatefulPartitionedCall:output:0dense_1_79101dense_1_79103*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_790602!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall^lstm_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2@
lstm_5/StatefulPartitionedCalllstm_5/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namelstm_3_input
?9
?
while_body_80295
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_3_matmul_readvariableop_resource_08
4while_lstm_cell_3_matmul_1_readvariableop_resource_07
3while_lstm_cell_3_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_3_matmul_readvariableop_resource6
2while_lstm_cell_3_matmul_1_readvariableop_resource5
1while_lstm_cell_3_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOp?
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul?
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	(?*
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOp?
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/MatMul_1?
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/add?
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOp?
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_3/BiasAddt
while/lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_3/Const?
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dim?
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
while/lstm_cell_3/split?
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid?
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid_1?
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul?
while/lstm_cell_3/ReluRelu while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Relu?
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0$while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul_1?
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/add_1?
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Sigmoid_2?
while/lstm_cell_3/Relu_1Reluwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/Relu_1?
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0&while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
while/lstm_cell_3/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0*
T0*'
_output_shapes
:?????????(2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????(:?????????(: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_78446
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_78446___redundant_placeholder03
/while_while_cond_78446___redundant_placeholder13
/while_while_cond_78446___redundant_placeholder23
/while_while_cond_78446___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_80294
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_80294___redundant_placeholder03
/while_while_cond_80294___redundant_placeholder13
/while_while_cond_80294___redundant_placeholder23
/while_while_cond_80294___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????(:?????????(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
:
?
?
+__inference_lstm_cell_5_layer_call_fn_82514

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_775372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????2:?????????2:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/1
?$
?
while_body_77353
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_4_77377_0
while_lstm_cell_4_77379_0
while_lstm_cell_4_77381_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_4_77377
while_lstm_cell_4_77379
while_lstm_cell_4_77381??)while/lstm_cell_4/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????(*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_77377_0while_lstm_cell_4_77379_0while_lstm_cell_4_77381_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_769272+
)while/lstm_cell_4/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1*^while/lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2*^while/lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_4_77377while_lstm_cell_4_77377_0"4
while_lstm_cell_4_77379while_lstm_cell_4_77379_0"4
while_lstm_cell_4_77381while_lstm_cell_4_77381_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?W
?
A__inference_lstm_5_layer_call_and_return_conditional_losses_82020

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????22
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	2?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_81935*
condR
while_cond_81934*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????2:::2
whilewhile:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?9
?
while_body_80951
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_4_matmul_readvariableop_resource_08
4while_lstm_cell_4_matmul_1_readvariableop_resource_07
3while_lstm_cell_4_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_4_matmul_readvariableop_resource6
2while_lstm_cell_4_matmul_1_readvariableop_resource5
1while_lstm_cell_4_biasadd_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????(*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	(?*
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/MatMul?
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp?
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/MatMul_1?
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/add?
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp?
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_4/BiasAddt
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const?
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
while/lstm_cell_4/split?
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid?
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid_1?
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul?
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Relu?
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul_1?
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/add_1?
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Sigmoid_2?
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/Relu_1?
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
while/lstm_cell_4/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?$
?
while_body_76611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_3_76635_0
while_lstm_cell_3_76637_0
while_lstm_cell_3_76639_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_3_76635
while_lstm_cell_3_76637
while_lstm_cell_3_76639??)while/lstm_cell_3/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_76635_0while_lstm_cell_3_76637_0while_lstm_cell_3_76639_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????(:?????????(:?????????(*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_762842+
)while/lstm_cell_3/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1*^while/lstm_cell_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2*^while/lstm_cell_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_3_76635while_lstm_cell_3_76635_0"4
while_lstm_cell_3_76637while_lstm_cell_3_76637_0"4
while_lstm_cell_3_76639while_lstm_cell_3_76639_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????(:?????????(: : :::2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????(:-)
'
_output_shapes
:?????????(:

_output_shapes
: :

_output_shapes
: 
?$
?
while_body_77963
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_5_77987_0
while_lstm_cell_5_77989_0
while_lstm_cell_5_77991_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_5_77987
while_lstm_cell_5_77989
while_lstm_cell_5_77991??)while/lstm_cell_5/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_5_77987_0while_lstm_cell_5_77989_0while_lstm_cell_5_77991_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????2:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_775372+
)while/lstm_cell_5/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_5/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_5/StatefulPartitionedCall:output:1*^while/lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_5/StatefulPartitionedCall:output:2*^while/lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_5_77987while_lstm_cell_5_77987_0"4
while_lstm_cell_5_77989while_lstm_cell_5_77989_0"4
while_lstm_cell_5_77991while_lstm_cell_5_77991_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2V
)while/lstm_cell_5/StatefulPartitionedCall)while/lstm_cell_5/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
: 
?V
?
A__inference_lstm_4_layer_call_and_return_conditional_losses_81364

inputs.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????22	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????(2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	(?*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/MatMul?
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp?
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/MatMul_1?
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/add?
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dim?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
	num_split2
lstm_cell_4/split?
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid?
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid_1?
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Relu?
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mul_1?
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/add_1?
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/Relu_1?
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????22
lstm_cell_4/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_81279*
condR
while_cond_81278*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(:::2
whilewhile:S O
+
_output_shapes
:?????????(
 
_user_specified_nameinputs
?V
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_80861

inputs.
*lstm_cell_3_matmul_readvariableop_resource0
,lstm_cell_3_matmul_1_readvariableop_resource/
+lstm_cell_3_biasadd_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????(2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????(2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!lstm_cell_3/MatMul/ReadVariableOp?
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul?
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	(?*
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp?
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/MatMul_1?
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/add?
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp?
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_3/BiasAddh
lstm_cell_3/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/Const|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dim?
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????(:?????????(:?????????(:?????????(*
	num_split2
lstm_cell_3/split?
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid?
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid_1?
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mulz
lstm_cell_3/ReluRelulstm_cell_3/split:output:2*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Relu?
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mul_1?
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/add_1?
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Sigmoid_2y
lstm_cell_3/Relu_1Relulstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/Relu_1?
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0 lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:?????????(2
lstm_cell_3/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????(:?????????(: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_80776*
condR
while_cond_80775*K
output_shapes:
8: : : : :?????????(:?????????(: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????(   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????(*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????(*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????(2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^while*
T0*+
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_78781
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_78781___redundant_placeholder03
/while_while_cond_78781___redundant_placeholder13
/while_while_cond_78781___redundant_placeholder23
/while_while_cond_78781___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_81431
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_81431___redundant_placeholder03
/while_while_cond_81431___redundant_placeholder13
/while_while_cond_81431___redundant_placeholder23
/while_while_cond_81431___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????2:?????????2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????2:-)
'
_output_shapes
:?????????2:

_output_shapes
: :

_output_shapes
:"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
lstm_3_input9
serving_default_lstm_3_input:0?????????;
dense_10
StatefulPartitionedCall:0?????????(tensorflow/serving/predict:??
?<
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"?:
_tf_keras_sequential?9{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_3_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 40, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTM", "config": {"name": "lstm_4", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_3_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 40, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTM", "config": {"name": "lstm_4", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["mean_squared_error"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_rnn_layer?
{"class_name": "LSTM", "name": "lstm_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 1]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 40, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 1]}}
?
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_4", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 40]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 40]}}
?
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 50]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 50]}}
?

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?
#iter

$beta_1

%beta_2
	&decay
'learning_ratemsmt(mu)mv*mw+mx,my-mz.m{/m|0m}v~v(v?)v?*v?+v?,v?-v?.v?/v?0v?"
	optimizer
 "
trackable_list_wrapper
n
(0
)1
*2
+3
,4
-5
.6
/7
08
9
10"
trackable_list_wrapper
n
(0
)1
*2
+3
,4
-5
.6
/7
08
9
10"
trackable_list_wrapper
?
regularization_losses
	variables

1layers
2metrics
3layer_regularization_losses
trainable_variables
4layer_metrics
5non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

(kernel
)recurrent_kernel
*bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_3", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
?
regularization_losses
	variables

:layers
;metrics
<layer_regularization_losses
trainable_variables
=layer_metrics

>states
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

+kernel
,recurrent_kernel
-bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_4", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
?
regularization_losses
	variables

Dlayers
Emetrics
Flayer_regularization_losses
trainable_variables
Glayer_metrics

Hstates
Inon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

.kernel
/recurrent_kernel
0bias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_5", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
?
regularization_losses
	variables

Nlayers
Ometrics
Player_regularization_losses
trainable_variables
Qlayer_metrics

Rstates
Snon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2(2dense_1/kernel
:(2dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
 	variables

Tlayers
Ulayer_regularization_losses
Vlayer_metrics
!trainable_variables
Wmetrics
Xnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*	?2lstm_3/lstm_cell_3/kernel
6:4	(?2#lstm_3/lstm_cell_3/recurrent_kernel
&:$?2lstm_3/lstm_cell_3/bias
,:*	(?2lstm_4/lstm_cell_4/kernel
6:4	2?2#lstm_4/lstm_cell_4/recurrent_kernel
&:$?2lstm_4/lstm_cell_4/bias
,:*	2?2lstm_5/lstm_cell_5/kernel
6:4	2?2#lstm_5/lstm_cell_5/recurrent_kernel
&:$?2lstm_5/lstm_cell_5/bias
<
0
1
2
3"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
?
6regularization_losses
7	variables

[layers
\layer_regularization_losses
]layer_metrics
8trainable_variables
^metrics
_non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
?
@regularization_losses
A	variables

`layers
alayer_regularization_losses
blayer_metrics
Btrainable_variables
cmetrics
dnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
?
Jregularization_losses
K	variables

elayers
flayer_regularization_losses
glayer_metrics
Ltrainable_variables
hmetrics
inon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	jtotal
	kcount
l	variables
m	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	ntotal
	ocount
p
_fn_kwargs
q	variables
r	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
j0
k1"
trackable_list_wrapper
-
l	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
n0
o1"
trackable_list_wrapper
-
q	variables"
_generic_user_object
%:#2(2Adam/dense_1/kernel/m
:(2Adam/dense_1/bias/m
1:/	?2 Adam/lstm_3/lstm_cell_3/kernel/m
;:9	(?2*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m
+:)?2Adam/lstm_3/lstm_cell_3/bias/m
1:/	(?2 Adam/lstm_4/lstm_cell_4/kernel/m
;:9	2?2*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m
+:)?2Adam/lstm_4/lstm_cell_4/bias/m
1:/	2?2 Adam/lstm_5/lstm_cell_5/kernel/m
;:9	2?2*Adam/lstm_5/lstm_cell_5/recurrent_kernel/m
+:)?2Adam/lstm_5/lstm_cell_5/bias/m
%:#2(2Adam/dense_1/kernel/v
:(2Adam/dense_1/bias/v
1:/	?2 Adam/lstm_3/lstm_cell_3/kernel/v
;:9	(?2*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v
+:)?2Adam/lstm_3/lstm_cell_3/bias/v
1:/	(?2 Adam/lstm_4/lstm_cell_4/kernel/v
;:9	2?2*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v
+:)?2Adam/lstm_4/lstm_cell_4/bias/v
1:/	2?2 Adam/lstm_5/lstm_cell_5/kernel/v
;:9	2?2*Adam/lstm_5/lstm_cell_5/recurrent_kernel/v
+:)?2Adam/lstm_5/lstm_cell_5/bias/v
?2?
G__inference_sequential_1_layer_call_and_return_conditional_losses_80173
G__inference_sequential_1_layer_call_and_return_conditional_losses_79716
G__inference_sequential_1_layer_call_and_return_conditional_losses_79107
G__inference_sequential_1_layer_call_and_return_conditional_losses_79077?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_1_layer_call_fn_80227
,__inference_sequential_1_layer_call_fn_79165
,__inference_sequential_1_layer_call_fn_80200
,__inference_sequential_1_layer_call_fn_79222?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_76211?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
lstm_3_input?????????
?2?
A__inference_lstm_3_layer_call_and_return_conditional_losses_80861
A__inference_lstm_3_layer_call_and_return_conditional_losses_80533
A__inference_lstm_3_layer_call_and_return_conditional_losses_80380
A__inference_lstm_3_layer_call_and_return_conditional_losses_80708?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_lstm_3_layer_call_fn_80544
&__inference_lstm_3_layer_call_fn_80883
&__inference_lstm_3_layer_call_fn_80555
&__inference_lstm_3_layer_call_fn_80872?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_lstm_4_layer_call_and_return_conditional_losses_81189
A__inference_lstm_4_layer_call_and_return_conditional_losses_81517
A__inference_lstm_4_layer_call_and_return_conditional_losses_81036
A__inference_lstm_4_layer_call_and_return_conditional_losses_81364?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_lstm_4_layer_call_fn_81539
&__inference_lstm_4_layer_call_fn_81200
&__inference_lstm_4_layer_call_fn_81211
&__inference_lstm_4_layer_call_fn_81528?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_lstm_5_layer_call_and_return_conditional_losses_82173
A__inference_lstm_5_layer_call_and_return_conditional_losses_81845
A__inference_lstm_5_layer_call_and_return_conditional_losses_81692
A__inference_lstm_5_layer_call_and_return_conditional_losses_82020?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_lstm_5_layer_call_fn_82184
&__inference_lstm_5_layer_call_fn_81856
&__inference_lstm_5_layer_call_fn_82195
&__inference_lstm_5_layer_call_fn_81867?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_82205?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_82214?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
7B5
#__inference_signature_wrapper_79259lstm_3_input
?2?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_82280
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_82247?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_lstm_cell_3_layer_call_fn_82314
+__inference_lstm_cell_3_layer_call_fn_82297?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_82380
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_82347?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_lstm_cell_4_layer_call_fn_82414
+__inference_lstm_cell_4_layer_call_fn_82397?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_82447
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_82480?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_lstm_cell_5_layer_call_fn_82497
+__inference_lstm_cell_5_layer_call_fn_82514?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_76211{()*+,-./09?6
/?,
*?'
lstm_3_input?????????
? "1?.
,
dense_1!?
dense_1?????????(?
B__inference_dense_1_layer_call_and_return_conditional_losses_82205\/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????(
? z
'__inference_dense_1_layer_call_fn_82214O/?,
%?"
 ?
inputs?????????2
? "??????????(?
A__inference_lstm_3_layer_call_and_return_conditional_losses_80380?()*O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "2?/
(?%
0??????????????????(
? ?
A__inference_lstm_3_layer_call_and_return_conditional_losses_80533?()*O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "2?/
(?%
0??????????????????(
? ?
A__inference_lstm_3_layer_call_and_return_conditional_losses_80708q()*??<
5?2
$?!
inputs?????????

 
p

 
? ")?&
?
0?????????(
? ?
A__inference_lstm_3_layer_call_and_return_conditional_losses_80861q()*??<
5?2
$?!
inputs?????????

 
p 

 
? ")?&
?
0?????????(
? ?
&__inference_lstm_3_layer_call_fn_80544}()*O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"??????????????????(?
&__inference_lstm_3_layer_call_fn_80555}()*O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"??????????????????(?
&__inference_lstm_3_layer_call_fn_80872d()*??<
5?2
$?!
inputs?????????

 
p

 
? "??????????(?
&__inference_lstm_3_layer_call_fn_80883d()*??<
5?2
$?!
inputs?????????

 
p 

 
? "??????????(?
A__inference_lstm_4_layer_call_and_return_conditional_losses_81036?+,-O?L
E?B
4?1
/?,
inputs/0??????????????????(

 
p

 
? "2?/
(?%
0??????????????????2
? ?
A__inference_lstm_4_layer_call_and_return_conditional_losses_81189?+,-O?L
E?B
4?1
/?,
inputs/0??????????????????(

 
p 

 
? "2?/
(?%
0??????????????????2
? ?
A__inference_lstm_4_layer_call_and_return_conditional_losses_81364q+,-??<
5?2
$?!
inputs?????????(

 
p

 
? ")?&
?
0?????????2
? ?
A__inference_lstm_4_layer_call_and_return_conditional_losses_81517q+,-??<
5?2
$?!
inputs?????????(

 
p 

 
? ")?&
?
0?????????2
? ?
&__inference_lstm_4_layer_call_fn_81200}+,-O?L
E?B
4?1
/?,
inputs/0??????????????????(

 
p

 
? "%?"??????????????????2?
&__inference_lstm_4_layer_call_fn_81211}+,-O?L
E?B
4?1
/?,
inputs/0??????????????????(

 
p 

 
? "%?"??????????????????2?
&__inference_lstm_4_layer_call_fn_81528d+,-??<
5?2
$?!
inputs?????????(

 
p

 
? "??????????2?
&__inference_lstm_4_layer_call_fn_81539d+,-??<
5?2
$?!
inputs?????????(

 
p 

 
? "??????????2?
A__inference_lstm_5_layer_call_and_return_conditional_losses_81692}./0O?L
E?B
4?1
/?,
inputs/0??????????????????2

 
p

 
? "%?"
?
0?????????2
? ?
A__inference_lstm_5_layer_call_and_return_conditional_losses_81845}./0O?L
E?B
4?1
/?,
inputs/0??????????????????2

 
p 

 
? "%?"
?
0?????????2
? ?
A__inference_lstm_5_layer_call_and_return_conditional_losses_82020m./0??<
5?2
$?!
inputs?????????2

 
p

 
? "%?"
?
0?????????2
? ?
A__inference_lstm_5_layer_call_and_return_conditional_losses_82173m./0??<
5?2
$?!
inputs?????????2

 
p 

 
? "%?"
?
0?????????2
? ?
&__inference_lstm_5_layer_call_fn_81856p./0O?L
E?B
4?1
/?,
inputs/0??????????????????2

 
p

 
? "??????????2?
&__inference_lstm_5_layer_call_fn_81867p./0O?L
E?B
4?1
/?,
inputs/0??????????????????2

 
p 

 
? "??????????2?
&__inference_lstm_5_layer_call_fn_82184`./0??<
5?2
$?!
inputs?????????2

 
p

 
? "??????????2?
&__inference_lstm_5_layer_call_fn_82195`./0??<
5?2
$?!
inputs?????????2

 
p 

 
? "??????????2?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_82247?()*??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????(
"?
states/1?????????(
p
? "s?p
i?f
?
0/0?????????(
E?B
?
0/1/0?????????(
?
0/1/1?????????(
? ?
F__inference_lstm_cell_3_layer_call_and_return_conditional_losses_82280?()*??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????(
"?
states/1?????????(
p 
? "s?p
i?f
?
0/0?????????(
E?B
?
0/1/0?????????(
?
0/1/1?????????(
? ?
+__inference_lstm_cell_3_layer_call_fn_82297?()*??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????(
"?
states/1?????????(
p
? "c?`
?
0?????????(
A?>
?
1/0?????????(
?
1/1?????????(?
+__inference_lstm_cell_3_layer_call_fn_82314?()*??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????(
"?
states/1?????????(
p 
? "c?`
?
0?????????(
A?>
?
1/0?????????(
?
1/1?????????(?
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_82347?+,-??}
v?s
 ?
inputs?????????(
K?H
"?
states/0?????????2
"?
states/1?????????2
p
? "s?p
i?f
?
0/0?????????2
E?B
?
0/1/0?????????2
?
0/1/1?????????2
? ?
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_82380?+,-??}
v?s
 ?
inputs?????????(
K?H
"?
states/0?????????2
"?
states/1?????????2
p 
? "s?p
i?f
?
0/0?????????2
E?B
?
0/1/0?????????2
?
0/1/1?????????2
? ?
+__inference_lstm_cell_4_layer_call_fn_82397?+,-??}
v?s
 ?
inputs?????????(
K?H
"?
states/0?????????2
"?
states/1?????????2
p
? "c?`
?
0?????????2
A?>
?
1/0?????????2
?
1/1?????????2?
+__inference_lstm_cell_4_layer_call_fn_82414?+,-??}
v?s
 ?
inputs?????????(
K?H
"?
states/0?????????2
"?
states/1?????????2
p 
? "c?`
?
0?????????2
A?>
?
1/0?????????2
?
1/1?????????2?
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_82447?./0??}
v?s
 ?
inputs?????????2
K?H
"?
states/0?????????2
"?
states/1?????????2
p
? "s?p
i?f
?
0/0?????????2
E?B
?
0/1/0?????????2
?
0/1/1?????????2
? ?
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_82480?./0??}
v?s
 ?
inputs?????????2
K?H
"?
states/0?????????2
"?
states/1?????????2
p 
? "s?p
i?f
?
0/0?????????2
E?B
?
0/1/0?????????2
?
0/1/1?????????2
? ?
+__inference_lstm_cell_5_layer_call_fn_82497?./0??}
v?s
 ?
inputs?????????2
K?H
"?
states/0?????????2
"?
states/1?????????2
p
? "c?`
?
0?????????2
A?>
?
1/0?????????2
?
1/1?????????2?
+__inference_lstm_cell_5_layer_call_fn_82514?./0??}
v?s
 ?
inputs?????????2
K?H
"?
states/0?????????2
"?
states/1?????????2
p 
? "c?`
?
0?????????2
A?>
?
1/0?????????2
?
1/1?????????2?
G__inference_sequential_1_layer_call_and_return_conditional_losses_79077w()*+,-./0A?>
7?4
*?'
lstm_3_input?????????
p

 
? "%?"
?
0?????????(
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_79107w()*+,-./0A?>
7?4
*?'
lstm_3_input?????????
p 

 
? "%?"
?
0?????????(
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_79716q()*+,-./0;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????(
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_80173q()*+,-./0;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????(
? ?
,__inference_sequential_1_layer_call_fn_79165j()*+,-./0A?>
7?4
*?'
lstm_3_input?????????
p

 
? "??????????(?
,__inference_sequential_1_layer_call_fn_79222j()*+,-./0A?>
7?4
*?'
lstm_3_input?????????
p 

 
? "??????????(?
,__inference_sequential_1_layer_call_fn_80200d()*+,-./0;?8
1?.
$?!
inputs?????????
p

 
? "??????????(?
,__inference_sequential_1_layer_call_fn_80227d()*+,-./0;?8
1?.
$?!
inputs?????????
p 

 
? "??????????(?
#__inference_signature_wrapper_79259?()*+,-./0I?F
? 
??<
:
lstm_3_input*?'
lstm_3_input?????????"1?.
,
dense_1!?
dense_1?????????(