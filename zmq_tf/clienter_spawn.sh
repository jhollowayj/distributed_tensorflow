for ((i=1;i<=8;i++));
do
    # echo $i
    gnome-terminal -x sh -c "python client_runner.py -vram 0.08 -grads 5"
    sleep 0.5
done

python client_runner.py -vram 0.08 -grads 5
