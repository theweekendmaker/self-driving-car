from random import randint

WIDTH = 700
HEIGHT = 600
SPEED = 4
car = Actor("racecar")
car.pos = 250,500
trackLeft = []
trackRight = []
trackCount = 0
trackPosition = 250
trackWidth = 120
trackDirection = False
gameStatus = 0

def draw():
    global gameStatus
    screen.fill((128,128,128))
    if gameStatus == 0:
        car.draw()
        b = 0
        while b < len(trackLeft):
            trackLeft[b].draw()
            trackRight[b].draw()
            b += 1
    if gameStatus == 1:
        # Red Flag
        screen.blit('rflag', (318, 268))
        #gameStatus = 0
    if gameStatus == 2:
        # Chequered Flag
        screen.blit('cflag', (318, 268))

def update():
    global gameStatus , trackCount
    if gameStatus == 0:
        if keyboard.left:
            car.x = car.x - 2
        if keyboard.right:
            car.x = car.x + 2
        updateTrack()
    if trackCount > 200: gameStatus = 2

def makeTrack(): # Function to make a new section of track
    global trackCount, trackLeft, trackRight, trackPosition, trackWidth
    trackLeft.append(Actor("barrier", pos = (trackPosition-trackWidth,0)))
    trackRight.append(Actor("barrier", pos = (trackPosition+trackWidth,0)))
    trackCount += 1

def updateTrack(): # Function to update where the track blocks appear
    global trackCount, trackPosition, trackDirection, trackWidth, gameStatus
    b = 0
    while b < len(trackLeft):
        if car.colliderect(trackLeft[b]) or car.colliderect(trackRight[b]):
            gameStatus = 1  # Red flag state
        trackLeft[b].y += SPEED
        trackRight[b].y += SPEED
        b += 1
    if trackLeft[0].y > 600:
        del trackLeft[0]
        del trackRight[0]
    if trackLeft[len(trackLeft)-1].y > 32:
        if trackDirection == False: trackPosition += 16
        if trackDirection == True: trackPosition -= 16
        if randint(0, 4) == 1: trackDirection = not trackDirection
        if trackPosition > 700-trackWidth: trackDirection = True
        if trackPosition < trackWidth: trackDirection = False
        makeTrack()

makeTrack()
