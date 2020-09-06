#!/bin/sh

cat /etc/passwd | grep "$USR" > /dev/null
if [[ $? -ne 0 ]]
then
  adduser -D $USR
  addgroup $USR $USR
fi
  python3 main.py
  chown $USR:$USR *
  rm -r __pycache__
exit 0
