# vim: filetype=sh

help='echo -e \n\tUsage: ./run.sh\n\t\t\t[-k|--kill]  - kill the current list of computers running\n\t\t\t[-c|--clean] - removes the logs from all computers\n'

COMMAND='run'
echo "$@"
# Parse command line parameters
for i in "$@"
do
  case $i in
    -k|--kill) COMMAND='kill'; shift ;;
    -c|--clean) COMMAND='clean'; shift ;;
    -r|--run) COMMAND='run'; shift ;; # Default
    --help) $help; exit ;;
  esac
done

echo $COMMAND

#####################################################################################
###          EDDIT THIS PART;  IT WILL FILL OUT THE FORMS FOR YOU                 ###
###    (SADLY, I'M NOT SURE IF YOU CAN DO MORE THAN ONE CUDA DEVICE AT A TIME.)   ###
#####################################################################################
PS_HOSTNAMES=( infinity )  #hatch naga ghost morita infinity reaper potts santaka )
PS_HOSTPORTS=( 2222 )      #2222 2222 2222 2222 2222 2222 2222 2222 )
WK_HOSTNAMES=( infinity )  #hatch naga ghost morita infinity reaper potts santaka )
WK_HOSTPORTS=( 2223 )      #2223 2223 2223 2223 2223 2223 2223 2223 )
WK_CVD_IDS=( 0 )           #0 0 0 0 0 0 0 )
#####################################################################################

#####################################################################################
CD="cd /mnt/pccfs/projects/distTF/modularDNN_Practice/"
RM_LOG="rm -r /mnt/pccfs/projects/distTF/modularDNN_Practice/log/"
KILL="fuser -k"
#####################################################################################
CVD="CUDA_VISIBLE_DEVICES"
FILE='main.py'
JN_PS="--job_name=ps"
JN_WK="--job_name=worker"
TID="--task_index"
#####################################################################################


##### BUILD PARAM SERVER STUFFS!!!
PS_FLAGGS='--ps_hosts='
SSH_PS=( ) # empty
for i in "${!PS_HOSTNAMES[@]}"; do 
  PS_FLAGGS="$PS_FLAGGS${PS_HOSTNAMES[$i]}:${PS_HOSTPORTS[$i]},"
  SSH_PS+=( "remote@${PS_HOSTNAMES[$i]}" )
done
PS_FLAGGS=${PS_FLAGGS::-1} # Removes last comma
#####

##### BUILD WORKER STUFFS!!!
WK_FLAGGS='--worker_hosts='
SSH_WK=( ) # empty
for i in "${!WK_HOSTNAMES[@]}"; do 
  WK_FLAGGS="$WK_FLAGGS${WK_HOSTNAMES[$i]}:${WK_HOSTPORTS[$i]},"
  SSH_WK+=( "remote@${WK_HOSTNAMES[$i]}" )
done
WK_FLAGGS=${WK_FLAGGS::-1} # Removes last comma
#####
COMBINED="$FILE $PS_FLAGGS $WK_FLAGGS"

###############################################
###############################################
###############################################

if [ "$COMMAND" = 'kill' ]; then
  echo "\n\tKilling servers"
  for i in "${!SSH_PS[@]}"; do 
    echo "$KILL ${PS_HOSTPORTS[$i]}/tcp;" | ssh ${SSH_PS[$i]} "bash -s"
  done
  # Kill WKs   ################################
  echo "\n\tKilling Workers"
  for i in "${!SSH_WK[@]}"; do
    echo "$KILL ${WK_HOSTPORTS[$i]}/tcp;" | ssh ${SSH_WK[$i]} "bash -s"
  done

###############################################
###############################################
###############################################

elif [ "$COMMAND" = 'clean' ]; then

  $RM_LOG # Since it's on the file server, we don't need to ssh around...
  
  # echo "\n\tCleaning servers"
  # for i in "${!SSH_PS[@]}"; do 
  #   echo "$RM_LOG" | ssh ${SSH_PS[$i]} "bash -s"
  # done
  # # Kill WKs   ################################
  # echo "\n\tCleaning Workers"
  # for i in "${!SSH_WK[@]}"; do
  #   echo "$RM_LOG" | ssh ${SSH_WK[$i]} "bash -s"
  # done

###############################################
###############################################
###############################################

elif [ "$COMMAND" = 'run' ]; then
  # LAUNCH PSs   ################################
  for i in "${!SSH_PS[@]}"; do 
    gnome-terminal -e "ssh ${SSH_PS[$i]} '$CD; $CVD='' python $COMBINED $JN_PS $TID=$i'"
    sleep 0.5
  done
  # LAUNCH WKs   ################################
  for i in "${!SSH_WK[@]}"; do 
    gnome-terminal -e "ssh ${SSH_WK[$i]} '$CD; $CVD=${WK_CVD_IDS[$i]} python $COMBINED $JN_WK $TID=$i'"
    sleep 0.5
  done
fi

###############################################
###############################################
###############################################

#  gnome-terminal -e "ssh ${SSH_WK[$i]} '$PREP ${WK_HOSTPORTS[$i]}/tcp; $CVD=${WK_CVD_IDS[$i]} python $COMBINED $JN_WK $TID=$i'"
#David    x.x.23.03   hatch
#Ben      x.x.23.35   naga
#Chris    x.x.23.36   ghost
#Jake     x.x.23.37   morita
#Jacob    x.x.23.38   infinity
#Connor   x.x.23.39   doctor
#Derek    x.x.23.40   tigerpaw
#Tylor    x.x.23.41   reaper
#Daniel   x.x.23.42   potts
#Robert   x.x.23.44   aji
#Kevin    x.x.23.45   santaka