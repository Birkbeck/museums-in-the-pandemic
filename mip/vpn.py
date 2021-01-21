# -*- coding: utf-8 -*-

""" 
VPN functions 



"""

VPN_SERVERS = ['uk-manchester','uk-london','uk-southampton','ireland','belgium',
                'isle-of-man','luxembourg','austria']

# %% Mac functions
def vpn_off_mac():
    ret = run_os_command('piactl disconnect')
    print(ret)

def vpn_on_mac():
    ret = run_os_command('piactl connect')
    print(ret)

def vpn_go_region_mac(reg):
    assert reg in VPN_SERVERS
    ret = run_os_command('piactl set region '+reg)
    print(">> vpn_go_region:",reg)
    return ret

def vpn_is_on_mac():
    ret = run_os_command('piactl get connectionstate') == 'Connected'
    return ret

# %% Windows functions
def vpn_random_region():
    vpn_go_region(random.choice(VPN_SERVERS))


# TODO: add commands for PIA on Windows

