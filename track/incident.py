import datetime
from InqscribeEvent import InqscribeEvent


class Incident:
    def __init__(self,start_time):
        self.start_time = start_time
        self.end_time = None
        self.duration = None
        self.inqscribe_events = []

    def prettyPrint(self):
            print('Type: {}'.format(type(self).__name__))
            print('\tStart time: {}'.format(str(self.start_time)))
            if(self.end_time):
                print('\End time: {}'.format(str(self.end_time)))
            else:
                print("\tERROR: interaction has no end.")
    def printInq(self):
            print('undefined implementation')

class TrialIndicent(Incident):
    def __init__(self,start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.inqscribe_events = []
        self.inqscribe_events.append(InqscribeEvent(start_time, 't', 'start'))
        self.inqscribe_events.append(InqscribeEvent(end_time, 't', 'ends'))
    def prettyPrint(self):
            print('Type: {}'.format(type(self).__name__))
            print('\tStart time: {}'.format(str(self.start_time)))
            if(self.end_time):
                print('\tEnd time: {}'.format(str(self.end_time)))
            else:
                print("\tERROR: interaction has no end.")

class BracketIncident(Incident):
    def __init__(self,start_time, beetleChar):
        self.beetleChar = beetleChar
        self.start_time = start_time
        self.inqscribe_events = []
        self.inqscribe_events.append(InqscribeEvent(start_time, self.beetleChar, 'fu.' + 's'))

    def endIncident(self, end_time):
        self.end_time = end_time
        self.inqscribe_events.append(InqscribeEvent(end_time, self.beetleChar, 'fu.' + 'e'))

    def prettyPrint(self):
            print('Type: {}'.format(type(self).__name__))
            print('\tBeetle: {}'.format(self.beetleChar))
            print('\tStart time: {}'.format(str(self.start_time)))
            if(self.end_time):
                print('\tEnd time: {}'.format(str(self.end_time)))
            else:
                print("\tERROR: interaction has no end.")

class StartIncident(Incident):
    def __init__(self,start_time,beetleChar):
        self.start_time = start_time
        self.inqscribe_events = []
        self.inqscribe_events.append(InqscribeEvent(start_time, beetleChar, 'first'))
    def prettyPrint(self):
        print('Type: {}'.format(type(self).__name__))
        print('\tStart time: {}'.format(str(self.start_time)))

class ProximityIncident(Incident):
    def __init__(self,start_time,white_avg_vel_in,black_avg_vel_in):
        self.start_time = start_time
        self.inqscribe_events = []
        if(black_avg_vel_in > white_avg_vel_in):
            self.initiator = "Black"
            self.initiator_confidence = black_avg_vel_in - white_avg_vel_in
        else:
            self.initiator = "White"
            self.initiator_confidence = white_avg_vel_in - black_avg_vel_in
        self.deinitiator = None
        self.deinitiator_confidence = None
        self.inqscribe_events.append(InqscribeEvent(start_time, self.initiator[:1].lower(), 'prox'))

    def endIncident(self, end_time, white_avg_vel_out, black_avg_vel_out):
        self.end_time = end_time
        if(black_avg_vel_out < white_avg_vel_out):
            self.deinitiator = "Black"
            self.deinitiator_confidence = white_avg_vel_out - black_avg_vel_out
        else:
            self.deinitiator = "White"
            self.deinitiator_confidence = black_avg_vel_out - white_avg_vel_out
        self.duration = self.end_time - self.start_time
        self.inqscribe_events.append(InqscribeEvent(end_time, self.deinitiator[:1].lower(), 'ends'))

    def prettyPrint(self):
            print('Type: {}'.format(type(self).__name__))
            print('\tStart time: {}'.format(str(self.start_time)))

            print('\tInitiator: {}'.format(self.initiator))
            print('\t\tConfidence: {}'.format(round(self.initiator_confidence,1)))
            if(self.end_time):
                print('\tEnd time: {}'.format(str(self.end_time)))
                print('\tDe-initiator: {}'.format(self.deinitiator))
                print('\t\tConfidence: {}'.format(round(self.deinitiator_confidence,1)))
                print('\tDuration: {}'.format(str(self.duration)))
            else:
                print("\tERROR: interaction has no end.")