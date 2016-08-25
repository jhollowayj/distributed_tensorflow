COMMAND='send_to_fsl';
for i in "$@"
do
  case $i in
    -down) COMMAND='send_to_pccl';  shift ;;
  esac
done


if [ "$COMMAND" = 'send_to_fsl' ]; then
    echo "Pushing up to FSL" 
    rsync -ru --progress  ./* jacobj66@ssh.fsl.byu.edu:/fslhome/jacobj66/fsl_groups/fslg_pccl/projects/modDNN/ --exclude '.git' --exclude '*.pyc' --exclude '__pycache__' --exclude '.idea' --exclude 'logs/*'
elif [ "$COMMAND" = 'send_to_pccl' ]; then
    echo "Pulling down from FSL"
    rsync -ru --progress jacobj66@ssh.fsl.byu.edu:/fslhome/jacobj66/fsl_groups/fslg_pccl/projects/modDNN/ ./ --exclude '.git' --exclude '*.pyc' --exclude '__pycache__' --exclude '.idea' --exclude 'logs/*'
fi