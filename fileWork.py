import warnings

class FileWork:
    
    def __init__(self, filename=None):
        """
        Lightweigth class for write/read from file and
        save information on hard disk
        
        Args:
            filename (str, default: None): filename of file 
                        to manipulate
        """
        if filename is not None:
            self.filename = filename
        
    def readFromFile(self, filename=None):
        """
        Func for read information from file
        
        Args:
            filename (str, default: None): filename of file to read
        """
        if filename is None and self.filename is None:
            raise ValueError("Filename didn't statement!")
        elif filename is None and self.filename is not None:
            filename = self.filename
        elif filename is not None and \
                self.filename is not None and \
                self.filename != filename:
            raise Warning("Filename state in class and state in function variable.\
                            In this case function variable is accept")
            
        with open(filename, 'r+') as f:
            data = f.read()
        return data
    
    def writeInFile(self, filename=None, message=None, append=True, concatN = True):
        """
        Func for write information in file. By default message is None,
        which means delete all information from file. If you want to 
        append information to the file you can turn on append boolean
        variable.

        CAUTION:
        For more easy use func concat string with '\n' Unicode symbol.
        If you want to shut down this module set 'concatN' to False
        
        Args:
            filename (str, default: None): filename of file to read
            message (str, default: None): message to write
            append (bool, default: False): append message to the file or not
            concatN (bool, default: True): concat '\n' to the message
        """
        if filename is None and self.filename is None:
            raise ValueError("Filename didn't statement!")
        elif filename is None and self.filename is not None:
            filename = self.filename
        elif filename is not None and \
                self.filename is not None and \
                self.filename != filename:
            raise Warning("Filename state in class and state in function variable.\
                            In this case function variable is accept")
        if append:
            with open(filename, 'a') as f:
                if concatN:
                    f.write(message + '\n')
                else:
                    f.write(message)
        else:
            with open(filename, 'w+') as f:
                if concatN:
                    f.write(message + '\n')
                else:
                    f.write(message)
                