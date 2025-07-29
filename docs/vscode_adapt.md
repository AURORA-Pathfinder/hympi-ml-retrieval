## Connecting to ADAPT w/ VSCode
2. Press `Control+Shift+P` to bring up command menu.
3. Type "Open SSH Configuration File" and select it.
4. Add this to that file when it opens and fill out the username:
```.
Host *
	PKCS11Provider="C:\Program Files\HID Global\ActivClient\acpkcs211.dll"
	User <your username>

Host adapt
    HostName adaptlogin.nccs.nasa.gov
```
5. Press `Control+Shift+P` to bring up command menu.
6. Type "Connect to Host" and select it, then choose 'adapt'
7. Follow the prompts and you should be connected!