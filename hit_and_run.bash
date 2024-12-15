#!/bin/bash 

if ! command -v $1 &> /dev/null
then
    echo "Installning $1"
    #uppdatera installerade paket
    sudo apt-get update > /dev/null
    #installera $1 
    sudo apt-get install $1 > /dev/null
fi

echo "$1 installed"
echo "Running $2 processes of $1 for $3 seconds in xterm"

for ((i=1; i<=$2; i++)); do
    # Kör programmet i ett xterm-fönster som bakgrundsprocess
    xterm -e $1 2> /dev/null &
done

sleep $3 

#döda alla xterm-processer
killall xterm

echo -n "Uninstall $1? (y/n)"
read val 

if [[ "$val" == "y" ]] 
then
    echo "Uninstallning $1"
    #avinstallera $1
    sudo apt-get remove $1
fi