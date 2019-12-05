#!/usr/bin/env python

#import blist
import random
import re
import sys

if len( sys.argv ) < 4:
	raise Exception( "Usage: mixer.py <reads> <stagger> <reads.txt>+ < <organisms.txt>" )
strReads, strStagger, astrReads = sys.argv[1], sys.argv[2], sys.argv[3:]
fStagger = int(strStagger) != 0
iReads = int(strReads)
#print(strStagger)
#print(iReads)
#print(sys.argv[3:])

hashReads = {}
for strReads in astrReads:
	pMatch = re.search( '^(?:.*\/)?(\S+?)(?:\.\S*)?$', strReads )
	if not pMatch:
		sys.stderr.write( "Illegal genome: %s\n" % strReads )
		continue
	hashReads[pMatch.group( 1 )] = strReads

#print(hashReads)
dTotal = 0
hashStagger = {}
hashGenomes = {}
astrReads = [] # blist.blist( [] )
astrProvenance = [] # blist.blist( [] )
for strLine in sys.stdin:
	strLine = strLine.strip( )
	if strLine[0] == "#":
		continue
	strOrg, strStagger = strLine.split( "\t" )
	strReads = hashReads.get( strOrg )
	if not strReads:
		sys.stderr.write( "Unknown genome: %s\n" % strOrg )
		continue
	#print(strReads)
	dStagger = float(strStagger) if fStagger else 1.0
	strRead = strID = ""
	iBegin = len( astrReads )
	for strRLine in open( strReads ):
		strRLine = strRLine.lstrip( )
		if strRLine[0] == ">":
			if len( strRead ) > 2:
				astrProvenance.append( strID )
				astrReads.append( strRead.strip( ) )
				#print( strRead.strip( ) )
			strID = strRLine[1:].strip( )
			strRead = ""
		else:
			strRead += strRLine
	if len( astrReads ) > iBegin:
		hashGenomes[strOrg] = (iBegin, len( astrReads ))
		hashStagger[strOrg] = dStagger
		dTotal += dStagger
		#print(strLine)
		#print(dStagger)
		#print(dTotal)
		#print( len( astrReads ))

astrOrgs = []
adOrgs = []
for strOrg, dStagger in hashStagger.items( ):
	astrOrgs.append( strOrg )
	adOrgs.append( dStagger / dTotal )

#print(astrOrgs)
#print(adOrgs)

iRead = 0
#print(iReads)
while( iRead < iReads ):
#	print(iRead)
	dOrg = random.random( )
	dSum = iOrg = 0
	for iOrg in range( len( astrOrgs ) ):
		dSum += adOrgs[iOrg]
		if dOrg <= dSum:
			break
#	print(astrOrgs[iOrg])
#	print(dOrg)
#	print(dSum)
	if iOrg < len( astrOrgs ):
		strOrg = astrOrgs[iOrg]
		iBegin, iEnd = hashGenomes[strOrg]
		iCur = random.randrange( iBegin, iEnd )
		strProv = astrProvenance[iCur]
		strRead = astrReads[iCur]
	else:
		strProv = strRead = ""
	print( ">R%09d %s" % (iRead, strProv) )
	print( strRead )
	iRead += 1
