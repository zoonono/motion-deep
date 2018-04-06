import sys

from PyQt5 import QtCore, QtWidgets
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from viewer_window import Ui_MainWindow
from data import *
from torchvision import transforms

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, test_dataset, pred_dataset):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        
        self.test_dataset = test_dataset
        self.pred_dataset = pred_dataset
        assert len(self.test_dataset) == len(self.pred_dataset), (
            "Lengths of data and predictions must be the same: " +
            str(len(self.test_dataset)) + ", " + str(len(self.pred_dataset)))
        
        self.corrupt = None # H x W x D
        self.truth = None # H x W x D
        self.pred = None # H x W x D
        self.loss = None # 1
        
        self.spinBox.setValue(0)
        self.spinBox_2.setValue(0)
        self.spinBox.valueChanged.connect(lambda x: self.draw_figure(self.spinBox.value()))
        self.spinBox_2.valueChanged.connect(lambda x: self.load_image(self.spinBox_2.value())
                                                   or self.draw_figure(self.spinBox.value()))
                                                   
        self.radioButton_axial.toggled.connect(lambda x: self.set_slice_max() 
                or self.draw_figure(self.spinBox.value(), new_image = True))
        self.radioButton_sagittal.toggled.connect(lambda x: self.set_slice_max() 
                or self.draw_figure(self.spinBox.value(), new_image = True))
        self.radioButton_coronal.toggled.connect(lambda x: self.set_slice_max() 
                or self.draw_figure(self.spinBox.value(), new_image = True))
        
        self.load_image(self.spinBox_2.value())
        
        self.slices = self.corrupt.shape[2]
        self.examples = len(self.test_dataset)
        self.spinBox.setMinimum(0)
        self.spinBox_2.setMinimum(0)
        self.spinBox.setMaximum(self.slices - 1)
        self.spinBox_2.setMaximum(self.examples - 1)
        
        dimensions = (781, 511)
        dpi = 0.95
        self.figure = Figure(dimensions, dpi = dpi)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self.frame)
        # truth-corr  pred-corr  corrupt
        # truth       pred       residual
        self.figure_truth_corrected = self.figure.add_subplot(231)
        self.figure_pred_corrected = self.figure.add_subplot(232)
        self.figure_corrupt = self.figure.add_subplot(233)
        
        self.figure_truth = self.figure.add_subplot(234)
        self.figure_pred = self.figure.add_subplot(235)
        self.figure_residual = self.figure.add_subplot(236)
        
        self.figure.subplots_adjust(top = 0.999, bottom = 0.001,
                                    left = 0.001, right = 0.999,
                                    wspace = 0.005, hspace = 0.005)
        
        self.draw_figure(self.spinBox.value(), new_image = True)
    
    def imshow(self, figure, img):
        i = figure.imshow(img, interpolation = 'nearest', cmap = 'gray', origin = 'lower')
        # i.set_clim(np.min(img), np.max(img))
        # i.set_clim(np.min(self.corrupt), np.max(self.corrupt)) # -1956.8088561356203 2168.8190891448953
        i.set_clim(-2000, 2000)
        return i
        
    def set_data(self, i, img):
        i.set_data(img)
        # i.set_clim(np.min(img), np.max(img))
    
    def draw_figure(self, slice, new_image = False):
        if self.radioButton_axial.isChecked():
            corrupt = self.corrupt[:,:,slice]
            truth = self.truth[:,:,slice]
            pred = self.pred[:,:,slice]
        elif self.radioButton_sagittal.isChecked():
            corrupt = self.corrupt[slice,:,:]
            truth = self.truth[slice,:,:]
            pred = self.pred[slice,:,:]
        elif self.radioButton_coronal.isChecked():
            corrupt = self.corrupt[:,slice,:]
            truth = self.truth[:,slice,:]
            pred = self.pred[:,slice,:]
        if new_image:
            self.image_truth_corrected = self.imshow(self.figure_truth_corrected, corrupt - truth)
            self.image_pred_corrected = self.imshow(self.figure_pred_corrected, corrupt - pred)
            self.image_corrupt = self.imshow(self.figure_corrupt, corrupt)
            self.image_truth = self.imshow(self.figure_truth, truth)
            self.image_pred = self.imshow(self.figure_pred, pred)
            self.image_residual = self.imshow(self.figure_residual, truth - pred)
        else:
            self.set_data(self.image_truth_corrected, corrupt - truth)
            self.set_data(self.image_pred_corrected, corrupt - pred)
            self.set_data(self.image_corrupt, corrupt)
            self.set_data(self.image_truth, truth)
            self.set_data(self.image_pred, pred)
            self.set_data(self.image_residual, truth - pred)
        self.canvas.draw()
        
        self.statusBar().showMessage(str(self.loss[0]))
    
    def set_slice_max(self):
        if self.radioButton_axial.isChecked():
            dim = 2
        elif self.radioButton_sagittal.isChecked():
            dim = 0
        elif self.radioButton_coronal.isChecked():
            dim = 1
        self.slices = self.corrupt.shape[dim]
        self.spinBox.setMaximum(self.slices - 1)
    
    def load_image(self, example):
        test = self.test_dataset[example]
        pred = self.pred_dataset[example]
        self.corrupt = test['image'] # H x W x D
        self.truth = test['label'] # H x W x D
        self.pred = pred['image'] # H x W x D
        self.loss = pred['label'] # 1 or D
        
        self.set_slice_max()
        
def main(): 
    a = QtWidgets.QApplication(sys.argv)
    
    filenames = GenericFilenames('../motion_data_resid_2d/', 'motion_corrupt_',
                                 'motion_resid_', '.npy', 8704)
    train_filenames, test_filenames = filenames.split((0.890625, 0.109375))
    test = MotionCorrDataset(test_filenames, lambda x: np.load(x), transform = BackDim(both = True))
    
    save_filenames = GenericFilenames('../dncnn/', 'motion_pred_dn_',
                             'motion_pred_loss_dn_', '.npy', 8704)
    train_save_filenames, test_save_filenames = save_filenames.split((0.890625, 0.109375))
    t = transforms.Compose([RemoveDims(), BackDim()])
    pred = MotionCorrDataset(test_save_filenames, lambda x: np.load(x), transform = t)
    
    # posts = ['7752to7821', '7821to7889', '7889to7957', 
             # '7957to8025', '8025to8057', '8057to8161']
    # test_filenames = PostFilenames('../dncnn_3d/', 'motion_corrupt_',
                              # 'motion_resid_', '.npy', posts)
    # pred_filenames = PostFilenames('../dncnn_3d/', 'motion_pred_dn_',
                              # 'motion_pred_loss_dn_', '.npy', posts)
    # test = MotionCorrDataset(test_filenames, lambda x: np.load(x))
    # pred = MotionCorrDataset(pred_filenames, lambda x: np.load(x))
    
    w = MainWindow(test, pred)
    w.show()
    sys.exit(a.exec_())
    
if __name__ == "__main__":
    main()
