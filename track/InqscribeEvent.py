import datetime


class InqscribeEvent:
    def __init__(self, time, identifier, activity):
        self.time = time
        self.identifier = identifier
        self.activity = activity

    def __str__(self):
        time_array = str(self.time).split(".", 2)
        formatted_time = time_array[0] + "."
        if len(time_array) > 1:
            formatted_time += (str(self.time).split(".", 2)[1])[:2]
        else:
            formatted_time += "00"
        if self.activity:
            return "[{}] {}.{}".format(formatted_time, self.identifier, self.activity)
        else:
            return "[{}] {}".format(formatted_time, self.identifier)

    def __lt__(self, other):
        return self.time < other.time

    def getTime(self):
        return self.time
