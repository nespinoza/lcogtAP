# -*- coding: utf-8 -*-
import numpy as np
import subprocess
import urllib
import pyfits
import jdcal
import shutil
import glob
import os
import argparse

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import requests
from bs4 import BeautifulSoup

import ast
from datetime import datetime
 
import smtplib
import mimetypes
from dateutil import parser
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

class Bimail:
	def __init__(self,subject,recipients):
		self.subject = subject
		self.recipients = recipients
		self.htmlbody = ''
		self.sender = "updated.results@gmail.com"
		self.senderpass = 'qaz123wsx456'
		self.attachments = []
 
	def send(self):
		msg = MIMEMultipart('alternative')
		msg['From']=self.sender
		msg['Subject']=self.subject
		msg['To'] = ", ".join(self.recipients) # to must be array of the form ['mailsender135@gmail.com']
		msg.preamble = "preamble goes here"
		#check if there are attachments if yes, add them
		if self.attachments:
			self.attach(msg)
		#add html body after attachments
		msg.attach(MIMEText(self.htmlbody, 'html'))
		#send
		s = smtplib.SMTP('smtp.gmail.com:587')
		s.starttls()
		s.login(self.sender,self.senderpass)
		s.sendmail(self.sender, self.recipients, msg.as_string())
		#test
		print msg
		s.quit()
	
	def htmladd(self, html):
		self.htmlbody = self.htmlbody+'<p></p>'+html
 
	def attach(self,msg):
                print self.attachments
		for f in self.attachments:
		
			ctype, encoding = mimetypes.guess_type(f)
			if ctype is None or encoding is not None:
				ctype = "application/octet-stream"
				
			maintype, subtype = ctype.split("/", 1)
 
			if maintype == "text":
				fp = open(f)
				# Note: we should handle calculating the charset
				attachment = MIMEText(fp.read(), _subtype=subtype)
				fp.close()
			elif maintype == "image":
				fp = open(f, "rb")
				attachment = MIMEImage(fp.read(), _subtype=subtype)
				fp.close()
			elif maintype == "audio":
				fp = open(f, "rb")
				attachment = MIMEAudio(fp.read(), _subtype=subtype)
				fp.close()
			else:
				fp = open(f, "rb")
				attachment = MIMEBase(maintype, subtype)
				attachment.set_payload(fp.read())
				fp.close()
				encoders.encode_base64(attachment)
			attachment.add_header("Content-Disposition", "attachment", filename=f.split('/')[-1])
			attachment.add_header('Content-ID', '<{0}>'.format(f.split('/')[-1]))
			msg.attach(attachment)
	
	def addattach(self, files):
		self.attachments = self.attachments + files

def datesok(year,month):
    if year == 2016:
        if month >= 10:
            return True
    elif year > 2016:
        return True
    return False

def login(URL, username=None, password=None):
    """
    authenticate to access HTTPS web service
    """
    s = requests.session()

    # Retrieve the CSRF token first
    s.get(URL, verify=False) # sets cookie
    csrftoken = s.cookies['csrftoken']

    login_data = dict(username=username, password=password,
        csrfmiddlewaretoken=csrftoken)
    s.post(URL, data=login_data, headers=dict(Referer=URL))
    return s

def get_login_data():
    return -1

def get_hs_coords(s,target):
    return -1

def spaced(input,space):
    fixed = False
    i = 0
    input = space+input
    while(not fixed):
        if(input[i:i+1] == '\n'):
           input = input[0:i+1]+space+input[i+1:]
           i = i + len(space)
        i = i + 1
        if(i == len(input)-1):
          fixed = True
    return input

def get_epic_coords(epicid):
    url = 'http://archive.stsci.edu/k2/epic/search.php?'
    url += 'action=Search'
    url += '&target='+epicid
    url += '&outputformat=CSV'
    lines = urllib.urlopen(url)
    data = {}
    counter = 0
    for line in lines:
        if counter == 0:
            names = line.split(',')
            names[-1] = names[-1].split('\n')[0]
            counter += 1
        elif counter == 1:
            dtypes = line.split(',')
            dtypes[-1] = dtypes[-1].split('\n')[0]
            counter += 1
        else:
            values = line.split(',')
            values[-1] = values[-1].split('\n')[0]
            for j in range(len(values)):
                if dtypes[j] == 'integer':
                    if values[j] != '':
                        data[names[j]] = int(values[j])
                    else:
                        data[names[j]] = -1
                elif dtypes[j] == 'float':
                    if values[j] != '':
                        data[names[j]] = float(values[j])
                    else:
                        data[names[j]] = -1
                else:
                    data[names[j]] = values[j]
    return data['RA'],data['Dec']

def get_general_coords(target):
    """
    Given a target name, returns RA and DEC from simbad.
    """
    try:
        url = "http://simbad.u-strasbg.fr/simbad/sim-id?Ident="+target+"&NbIdent=1&Radius=2&Radius.unit=arcmin&submit=submit+id"
        urllib.urlopen(url)
        html = urlopen(url).read()
        splt = html.split('ICRS')
        rahh,ramm,rass,decdd,decmm,decss = (splt[1].split('<TT>\n')[1].split('\n')[0]).split()
        return rahh+':'+ramm+':'+rass,decdd+':'+decmm+':'+decss
    except:
        coords_file = open('../manual_object_coords.dat','r')
        while True:
            line = coords_file.readline()
            if line != '':
                name,ra,dec = line.split()
                if name.lower() == target.lower():
                    coords_file.close()
                    return ra,dec
            else:
                break
        coords_file.close()
        return 'NoneFound','NoneFound'

# Get user input:
parserIO = argparse.ArgumentParser()
parserIO.add_argument('-project',default=None)
args = parserIO.parse_args()

# Get the project name (see the userdata.dat file):
project = args.project

# Check for which project the user whishes to download data from:
fprojects = open('../userdata.dat','r')
while True:
    line = fprojects.readline()
    if line != '':
        if line[0] != '#':
            cp,cf = line.split()
            cp = cp.split()[0]
            cf = cf.split()[0]
            if project.lower() == cp.lower():
                break
    else:
        print '\t > Project '+project+' is not on the list of saved projects. '
        print '\t   Please associate it on the userdata.dat file.'

data_folder = cf
sendemail = False
emails_to_send = ['your.email@gmail.com']

folders_raw = glob.glob(data_folder+'LCOGT/raw/*')
dates_raw = len(folders_raw)*[[]]
for i in range(len(folders_raw)):
    dates_raw[i] = folders_raw[i].split('/')[-1]
folders_red = glob.glob(data_folder+'LCOGT/red/*')
dates_red = len(folders_red)*[[]]
for i in range(len(folders_red)):
    dates_red[i] = folders_red[i].split('/')[-1]

# Run the get_photometry_lcogt code for all the raw folders in case new data from past nights was 
# reduced by LCOGT today. If no new data, nothing will happen (i.e., the code does nothing):
today_jd = sum(jdcal.gcal2jd(str(datetime.today().year), str(datetime.today().month), str(datetime.today().day)))
for i in range(len(dates_raw)):
    first_HS_login = True
    year = int(dates_raw[i][:4])
    month = int(dates_raw[i][4:6])
    day = int(dates_raw[i][6:8])
    s = str(year)+'.'+str(month)+'.'+str(day)
    dt = parser.parse(s)
    data_jd = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day))
    # Only check data one week appart (maximum LCOGT takes to process data is ~couple of days, but one week
    # is the limit just to be sure):
    if data_jd > today_jd-7.:
        # Get already reduced targets (if any):
        bf = glob.glob(data_folder+'LCOGT/red/'+dates_raw[i]+'/*')
        before_target_folders = []
        for tar_dir in bf:
            if os.path.exists(tar_dir+'/sinistro'):
                before_target_folders.append(tar_dir)
        # Reduce the data (if already reduced, nothing will happen):
        print 'Reducing data for '+dates_raw[i]+' night. Reducing...'
        os.system('python get_photometry_lcogt.py -project HATSOUTH -datafolder '+dates_raw[i]) 
        # Now, assuming it is done, run the post-processing. First, switch to the post-processing folder:
        cwd = os.getcwd()
        os.chdir('../post_processing')
        out_folder = data_folder+'LCOGT/red/'+dates_raw[i]+'/'
        target_folders = glob.glob(out_folder+'*')
        # First, go through every observed object for the given night:
        for target_folder in target_folders:
          # Post-process the target only if it has already not been done:
          if target_folder not in before_target_folders:
            target = target_folder.split('/')[-1]
            print 'Post-processing target '+target+' on folder '+target_folder
            # If it is a HATS target, query to canditrack, get RA and DEC, and run the 
            # post-processing algorithm. If not, assume it is a K2 object whose EPIC
            # number is in 'target' (e.g., target = '214323253242-ip'). In that case, 
            # query RA and DEC from MAST:
            if target[0:4] == 'HATS':
                if 'Inter-American' in target:
                    name,number,band,n1,n2 = target.split('-')
                    dome = n1+'-'+n2
                else:
                    name,number,band,dome = target.split('-')
                target_name = '-'.join([name,number])
                if first_HS_login:
                    s = get_login_data()
                    first_HS_login = False
                try:
                    RA,DEC = get_hs_coords(s,target_name)
                    print '\t Found RA and DEC:',RA,DEC
                    targetok = True
                except:
                    print '\t RA and DEC obtention failed!'
                    targetok = False
            else:
                if 'Inter-American' in target:
                    splitted_name = target.split('-')
                    n2 = splitted_name[-1]
                    n1 = splitted_name[-2]
                    band = splitted_name[-3]
                    target_name = '-'.join(splitted_name[:-3])
                    dome = n1+'-'+n2
                else:
                    splitted_name = target.split('-')
                    dome = splitted_name[-1]
                    band = splitted_name[-2]
                    target_name = '-'.join(splitted_name[:-2])
                try:
                    RA,DEC = get_epic_coords(target_name)
                    RA = ':'.join(RA.split())
                    DEC = ':'.join(DEC.split())
                    print '\t Found RA and DEC:',RA,DEC
                    targetok = True
                except:
                    RA,DEC = get_general_coords(target_name)
                    if RA == 'NoneFound':
                        targetok = False
                    else:
                        targetok = True
            # Assuming RA an DEC have been retrieved, run the post-processing algorithm:
            if targetok:
              for ap in ['opt','5','15','20','30']:
                p = subprocess.Popen('echo $DISPLAY',stdout = subprocess.PIPE, \
                           stderr = subprocess.PIPE,shell = True)
                p.wait()
                out, err = p.communicate()
                if ap == 'opt':
                    code = 'python transit_photometry.py -project HATSOUTH -datafolder '+\
                           dates_raw[i]+' -target_name '+target_name+' -band '+band+\
                           ' -dome '+dome+' -ra "'+RA+'" -dec " '+DEC+'" -ncomp 10 --plt_images --autosaveLC'
                else:
                    code = 'python transit_photometry.py -project HATSOUTH -datafolder '+\
                           dates_raw[i]+' -target_name '+target_name+' -band '+band+\
                           ' -dome '+dome+' -ra "'+RA+'" -dec " '+DEC+'" -ncomp 10 --plt_images --force_aperture -forced_aperture '+ap+' --autosaveLC'
                print code
                p = subprocess.Popen(code,stdout = subprocess.PIPE, \
                           stderr = subprocess.PIPE,shell = True)
                p.wait()
                if sendemail:
                    if(p.returncode != 0 and p.returncode != None):
                        print 'Error sending mail:'
                        out, err = p.communicate()
                        print spaced(err,"\t \t")
                    print 'Sending e-mail...' 
                    mymail = Bimail('LCOGT DR: '+target_name+' on ' +dates_raw[i]+' Aperture: '+ap, emails_to_send)
                    mymail.htmladd('Data reduction was a SUCCESS! Attached is the lightcurve data.')
                    out = glob.glob(data_folder+'LCOGT/red/'+dates_raw[i]+'/'+target+'/*')
                    for ii in range(len(out)):
                       if out[ii].split('/')[-1] == 'sinistro':
                           out_folder = out[ii]
                           camera = 'sinistro'
                           break
                       elif out[ii].split('/')[-1] == 'sbig':
                           out_folder = out[ii]
                           camera = 'SBIG'
                           break
                    shutil.move(out_folder,out_folder+'_'+ap)
                    out_folder = out_folder+'_'+ap
                    real_camera = 'sinistro' # from now on, all LCOGT data comes from sinistro cameras
                    imgs = glob.glob(out_folder+'/target/*')
                    d,h = pyfits.getdata(data_folder+'LCOGT/raw/'+dates_raw[i]+'/'+(imgs[0].split('/')[-1]).split('.')[0]+'.fits',header=True)
                    mymail.htmladd('Camera: '+camera)
                    mymail.htmladd('Observing site: '+h['SITE'])
                    mymail.htmladd('Band: '+band)
                    mymail.htmladd('Dome: '+dome)
                    if len(imgs)>2:
                        mymail.addattach([imgs[0]])
                        mymail.addattach([imgs[1]])
                        mymail.addattach([imgs[2]])
                    elif len(imgs)==2:
                        mymail.addattach([imgs[0]])
                        mymail.addattach([imgs[1]])
                    else:
                        mymail.addattach([imgs[0]])
                    mymail.addattach([out_folder+'/'+target_name+'.dat'])
                    mymail.addattach([out_folder+'/'+target_name+'.pdf'])
                    mymail.addattach([out_folder+'/LC/'+target_name+'.epdlc'])
                    mymail.send()
                    shutil.move(out_folder[:-3]+'_opt',data_folder+'LCOGT/red/'+dates_raw[i]+'/'+target+'/sinistro')
                else:
                    mymail = Bimail('LCOGT DR: '+target_name+' on ' +datetime.now().strftime('%Y/%m/%d'), emails_to_send)
                    mymail.htmladd('Post-processing failed for object '+target+' on '+dates_raw[i])
                    mymail.send()
        # Get back to photometric pipeline directory:
        os.chdir(cwd) 
