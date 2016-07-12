# vim: filetype=sh

help='echo -e \n\tUsage: ./run.sh\n\t\t\t[-k|--kill]  - kill the current list of computers running\n\t\t\t[-c|--clean] - removes the logs from all computers\n'

COMMAND='run'
# echo "$@"
# Parse command line parameters
for i in "$@"
do
  case $i in
    -d|--debug) COMMAND='debug'; shift ;;
    -a|--all) COMMAND='all'; shift ;;
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
PS_HOSTNAMES=( morita hatch naga ghost infinity reaper potts santaka )
PS_HOSTPORTS=( 2222   2222  2222 2222  2222     2222   2222  2222 )
WK_HOSTNAMES_TRAIN=( morita infinity reaper ghost santaka potts naga hatch  morita infinity reaper ghost santaka potts naga hatch  morita infinity )
WK_HOSTPORTS_TRAIN=( 2223   2223     2223   2223  2223    2223  2223 2223   2224   2224     2224   2224  2224    2224  2224 2224   2225   2225     )
WK_WAT_IDS_TRAIN=("--world_id=1 --task_id=1 --agent_id=1" 
                  "--world_id=1 --task_id=1 --agent_id=2"
                  "--world_id=1 --task_id=2 --agent_id=1"
                  "--world_id=1 --task_id=2 --agent_id=3"
                  "--world_id=1 --task_id=3 --agent_id=2"
                  "--world_id=1 --task_id=3 --agent_id=3"
                  "--world_id=2 --task_id=1 --agent_id=1"
                  "--world_id=2 --task_id=1 --agent_id=3"
                  "--world_id=2 --task_id=2 --agent_id=2"
                  "--world_id=2 --task_id=2 --agent_id=3"
                  "--world_id=2 --task_id=3 --agent_id=1"
                  "--world_id=2 --task_id=3 --agent_id=2"
                  "--world_id=3 --task_id=1 --agent_id=2"
                  "--world_id=3 --task_id=1 --agent_id=3"
                  "--world_id=3 --task_id=2 --agent_id=1"
                  "--world_id=3 --task_id=2 --agent_id=2"
                  "--world_id=3 --task_id=3 --agent_id=1"
                  "--world_id=3 --task_id=3 --agent_id=3" )

WK_HOSTNAMES_EVAL=( morita reaper ghost santaka potts naga hatch  infinity reaper)
WK_HOSTPORTS_EVAL=( 2230   2230   2230  2230    2230  2230 2230   2230     2231)
WK_WAT_IDS_EVAL=("--world_id=1 --task_id=1 --agent_id=3 --observer=True" 
                 "--world_id=1 --task_id=2 --agent_id=2 --observer=True"
                 "--world_id=1 --task_id=3 --agent_id=1 --observer=True"
                 "--world_id=2 --task_id=1 --agent_id=2 --observer=True"
                 "--world_id=2 --task_id=2 --agent_id=1 --observer=True"
                 "--world_id=2 --task_id=3 --agent_id=3 --observer=True"
                 "--world_id=3 --task_id=1 --agent_id=1 --observer=True"
                 "--world_id=3 --task_id=2 --agent_id=3 --observer=True"
                 "--world_id=3 --task_id=3 --agent_id=1 --observer=True" )
#####################################################################################

#####################################################################################
CD="cd /mnt/pccfs/projects/distTF/modularDNN_Practice/"
RM_LOG="rm -r /mnt/pccfs/projects/distTF/modularDNN_Practice/logs/"
KILL="fuser -k" # [portnum]/tcp;
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

WK_FLAGGS='--worker_hosts='
SSH_WK=( ) # empty
WK_WorldAgentTaskIDs=( ) # empty
WK_HOSTPORTS_COMBINED=( ) # empty
##### BUILD WORKER (TRAIN)
for i in "${!WK_HOSTNAMES_TRAIN[@]}"; do 
  WK_FLAGGS="$WK_FLAGGS${WK_HOSTNAMES_TRAIN[$i]}:${WK_HOSTPORTS_TRAIN[$i]},"
  SSH_WK+=( "remote@${WK_HOSTNAMES_TRAIN[$i]}" )
  WK_WorldAgentTaskIDs+=( "${WK_WAT_IDS_TRAIN[$i]}" )
  WK_HOSTPORTS_COMBINED+=( ${WK_HOSTPORTS_TRAIN[$i]} )
done

##### BUILD WORKER (EVAL)
for i in "${!WK_HOSTNAMES_EVAL[@]}"; do 
  WK_FLAGGS="$WK_FLAGGS${WK_HOSTNAMES_EVAL[$i]}:${WK_HOSTPORTS_EVAL[$i]},"
  SSH_WK+=( "remote@${WK_HOSTNAMES_EVAL[$i]}" )
  WK_WorldAgentTaskIDs+=( "${WK_WAT_IDS_EVAL[$i]}" )
  WK_HOSTPORTS_COMBINED+=( ${WK_HOSTPORTS_EVAL[$i]} )
done
#####
WK_FLAGGS=${WK_FLAGGS::-1} # Removes last comma
COMBINED="$FILE $PS_FLAGGS $WK_FLAGGS"

###############################################
###############################################
###############################################

if [ "$COMMAND" = 'debug' ]; then
  echo "=-=-=-=-=-=-=-==-==-=-=-=-=-"
  echo "=-=-=-=-= KILL =-==-=-=-=-=-"
  echo "=-=-=-=-=-=-=-==-==-=-=-=-=-"
  echo "\n\tKilling servers"
  for i in "${!SSH_PS[@]}"; do 
    echo "$KILL ${PS_HOSTPORTS[$i]}/tcp; | ssh ${SSH_PS[$i]} \"bash -s\""
  done
  # Kill WKs   ################################
  echo "\n\tKilling Workers"
  for i in "${!SSH_WK[@]}"; do
    echo "$KILL ${WK_HOSTPORTS_COMBINED[$i]}/tcp; | ssh ${SSH_WK[$i]} \"bash -s\""
  done
  echo "=-=-=-=-=-=-=-==-==-=-=-=-=-"
  echo "=-=-=-=-= CLEAN -==-=-=-=-=-"
  echo "=-=-=-=-=-=-=-==-==-=-=-=-=-"
  echo $RM_LOG 
  echo "=-=-=-=-=-=-=-==-==-=-=-=-=-"
  echo "=-=-=-=-= CLEAN -==-=-=-=-=-"
  echo "=-=-=-=-=-=-=-==-==-=-=-=-=-"
  for i in "${!SSH_PS[@]}"; do 
    echo "gnome-terminal -e \"ssh ${SSH_PS[$i]} '$CD; $CVD='' python $COMBINED $JN_PS $TID=$i'\""
  done
  # LAUNCH WKs   ################################
  for i in "${!SSH_WK[@]}"; do 
    # gnome-terminal -e "ssh ${SSH_WK[$i]} '$CD; $CVD=${WK_CVD_IDS[$i]} python $COMBINED $JN_WK $TID=$i ${WK_WAT[$i]}'" # GPU enabled
    echo "gnome-terminal -e \"ssh ${SSH_WK[$i]} '$CD; $CVD='' python $COMBINED $JN_WK $TID=$i ${WK_WAT[$i]} ${WK_WorldAgentTaskIDs[$i]}'\"" # CPU ONLY
  done

###############################################
###############################################
###############################################

elif [ "$COMMAND" = 'all' ]; then
  bash screenplay.sh -k
  bash screenplay.sh -c
  bash screenplay.sh -r

###############################################
###############################################
###############################################

elif [ "$COMMAND" = 'kill' ]; then
  echo "\n\tKilling servers"
  for i in "${!SSH_PS[@]}"; do 
    echo "$KILL ${PS_HOSTPORTS[$i]}/tcp;" | ssh ${SSH_PS[$i]} "bash -s"
  done
  # Kill WKs   ################################
  echo "\n\tKilling Workers"
  for i in "${!SSH_WK[@]}"; do
    echo "$KILL ${PS_HOSTPORTS[$i]}/tcp;" | ssh ${SSH_PS[$i]} "bash -s"
    echo "$KILL ${WK_HOSTPORTS_COMBINED[$i]}/tcp;" | ssh ${SSH_WK[$i]} "bash -s"
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
    # gnome-terminal -e "ssh ${SSH_WK[$i]} '$CD; $CVD=${WK_CVD_IDS[$i]} python $COMBINED $JN_WK $TID=$i ${WK_WAT[$i]}'" # GPU enabled
    gnome-terminal -e "ssh ${SSH_WK[$i]} '$CD; $CVD='' python $COMBINED $JN_WK $TID=$i ${WK_WAT[$i]} ${WK_WorldAgentTaskIDs[$i]}'" # CPU ONLY
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