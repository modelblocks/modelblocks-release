import sys

## NOTE: Utterance ID and position are output as sentid and sentpos because downstream scripts require those fields
uttid = 0
## rate is words / second
print('word sentid sentpos timestamp rate')
with open(sys.argv[1],'r') as f:
  for line in f:
    times,utterance = line.strip().split("''",1)
    utterance = utterance.strip("''")
    start,stop = times.strip(', ').split(',')
    start = float(start)
    stop = float(stop)
    words = utterance.split()
    uttpos = 1
    timestamp = start
    timestep = (stop-start)/len(words)
    for word in words:
      ## NOTE: timestamp **assumes a uniform speech rate over each utterance**
      ## This is a ridiculous assumption, which was mainly made for piloting
      print word, uttid, uttpos, timestamp, 1/timestep
      timestamp += timestep
      uttpos += 1
    uttid += 1
