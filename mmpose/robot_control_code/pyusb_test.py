import usb.core

vid = 0x0525
pid = 0xA4AC
'''
dev = usb.core.find(idVendor=vid, idProduct=pid)
if dev is None:
    raise ValueError('Our device is not connected')
'''
all_devs = usb.core.find(find_all=True)
print(all_devs)
for d in all_devs:
    if(d.idVendor == vid) & (d.idProduct == pid):
        print(d)
