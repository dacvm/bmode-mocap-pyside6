# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'volume.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSlider, QVBoxLayout,
    QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(600, 800)
        self.verticalLayout_2 = QVBoxLayout(Form)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.gridLayout_volume_menu = QGridLayout()
        self.gridLayout_volume_menu.setObjectName(u"gridLayout_volume_menu")
        self.lineEdit_volume_volfile = QLineEdit(Form)
        self.lineEdit_volume_volfile.setObjectName(u"lineEdit_volume_volfile")

        self.gridLayout_volume_menu.addWidget(self.lineEdit_volume_volfile, 0, 5, 1, 1)

        self.pushButton_volume_configfileClear = QPushButton(Form)
        self.pushButton_volume_configfileClear.setObjectName(u"pushButton_volume_configfileClear")
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.EditClear))
        self.pushButton_volume_configfileClear.setIcon(icon)

        self.gridLayout_volume_menu.addWidget(self.pushButton_volume_configfileClear, 0, 2, 1, 1)

        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout_volume_menu.addWidget(self.label_3, 2, 0, 1, 1)

        self.pushButton_volume_outputdirClear = QPushButton(Form)
        self.pushButton_volume_outputdirClear.setObjectName(u"pushButton_volume_outputdirClear")
        self.pushButton_volume_outputdirClear.setIcon(icon)

        self.gridLayout_volume_menu.addWidget(self.pushButton_volume_outputdirClear, 2, 2, 1, 1)

        self.lineEdit_volume_outputdir = QLineEdit(Form)
        self.lineEdit_volume_outputdir.setObjectName(u"lineEdit_volume_outputdir")

        self.gridLayout_volume_menu.addWidget(self.lineEdit_volume_outputdir, 2, 1, 1, 1)

        self.pushButton_volume_seqfileClear = QPushButton(Form)
        self.pushButton_volume_seqfileClear.setObjectName(u"pushButton_volume_seqfileClear")
        self.pushButton_volume_seqfileClear.setIcon(icon)

        self.gridLayout_volume_menu.addWidget(self.pushButton_volume_seqfileClear, 1, 2, 1, 1)

        self.label = QLabel(Form)
        self.label.setObjectName(u"label")

        self.gridLayout_volume_menu.addWidget(self.label, 0, 0, 1, 1)

        self.lineEdit_volume_configfile = QLineEdit(Form)
        self.lineEdit_volume_configfile.setObjectName(u"lineEdit_volume_configfile")

        self.gridLayout_volume_menu.addWidget(self.lineEdit_volume_configfile, 0, 1, 1, 1)

        self.lineEdit_volume_seqfile = QLineEdit(Form)
        self.lineEdit_volume_seqfile.setObjectName(u"lineEdit_volume_seqfile")

        self.gridLayout_volume_menu.addWidget(self.lineEdit_volume_seqfile, 1, 1, 1, 1)

        self.pushButton_volume_volfileBrowse = QPushButton(Form)
        self.pushButton_volume_volfileBrowse.setObjectName(u"pushButton_volume_volfileBrowse")
        icon1 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.DocumentOpen))
        self.pushButton_volume_volfileBrowse.setIcon(icon1)

        self.gridLayout_volume_menu.addWidget(self.pushButton_volume_volfileBrowse, 0, 6, 1, 1)

        self.pushButton_volume_seqfileBrowse = QPushButton(Form)
        self.pushButton_volume_seqfileBrowse.setObjectName(u"pushButton_volume_seqfileBrowse")
        self.pushButton_volume_seqfileBrowse.setIcon(icon1)

        self.gridLayout_volume_menu.addWidget(self.pushButton_volume_seqfileBrowse, 1, 3, 1, 1)

        self.pushButton_volume_outputdirBrowse = QPushButton(Form)
        self.pushButton_volume_outputdirBrowse.setObjectName(u"pushButton_volume_outputdirBrowse")
        icon2 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.FolderOpen))
        self.pushButton_volume_outputdirBrowse.setIcon(icon2)

        self.gridLayout_volume_menu.addWidget(self.pushButton_volume_outputdirBrowse, 2, 3, 1, 1)

        self.label_7 = QLabel(Form)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout_volume_menu.addWidget(self.label_7, 1, 4, 1, 1)

        self.pushButton_volume_reconstruct = QPushButton(Form)
        self.pushButton_volume_reconstruct.setObjectName(u"pushButton_volume_reconstruct")

        self.gridLayout_volume_menu.addWidget(self.pushButton_volume_reconstruct, 3, 0, 1, 4)

        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_volume_menu.addWidget(self.label_2, 1, 0, 1, 1)

        self.pushButton_volume_configfileBrowse = QPushButton(Form)
        self.pushButton_volume_configfileBrowse.setObjectName(u"pushButton_volume_configfileBrowse")
        self.pushButton_volume_configfileBrowse.setIcon(icon1)

        self.gridLayout_volume_menu.addWidget(self.pushButton_volume_configfileBrowse, 0, 3, 1, 1)

        self.label_4 = QLabel(Form)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_volume_menu.addWidget(self.label_4, 0, 4, 1, 1)

        self.pushButton_volume_volload = QPushButton(Form)
        self.pushButton_volume_volload.setObjectName(u"pushButton_volume_volload")

        self.gridLayout_volume_menu.addWidget(self.pushButton_volume_volload, 0, 7, 1, 1)

        self.horizontalSlider_volume_threshold = QSlider(Form)
        self.horizontalSlider_volume_threshold.setObjectName(u"horizontalSlider_volume_threshold")
        self.horizontalSlider_volume_threshold.setOrientation(Qt.Orientation.Horizontal)
        self.horizontalSlider_volume_threshold.setTickPosition(QSlider.TickPosition.TicksBothSides)

        self.gridLayout_volume_menu.addWidget(self.horizontalSlider_volume_threshold, 1, 5, 1, 3)


        self.verticalLayout.addLayout(self.gridLayout_volume_menu)

        self.widget_volume_scatter = QWidget(Form)
        self.widget_volume_scatter.setObjectName(u"widget_volume_scatter")
        self.verticalLayout_4 = QVBoxLayout(self.widget_volume_scatter)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_volume_scatter = QVBoxLayout()
        self.verticalLayout_volume_scatter.setObjectName(u"verticalLayout_volume_scatter")

        self.verticalLayout_4.addLayout(self.verticalLayout_volume_scatter)


        self.verticalLayout.addWidget(self.widget_volume_scatter)

        self.verticalLayout.setStretch(1, 3)

        self.verticalLayout_2.addLayout(self.verticalLayout)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.lineEdit_volume_volfile.setPlaceholderText(QCoreApplication.translate("Form", u"~/volume.mha", None))
        self.pushButton_volume_configfileClear.setText("")
        self.label_3.setText(QCoreApplication.translate("Form", u"Output Dir", None))
        self.pushButton_volume_outputdirClear.setText("")
        self.lineEdit_volume_outputdir.setPlaceholderText(QCoreApplication.translate("Form", u"~/output/", None))
        self.pushButton_volume_seqfileClear.setText("")
        self.label.setText(QCoreApplication.translate("Form", u"Config File", None))
        self.lineEdit_volume_configfile.setPlaceholderText(QCoreApplication.translate("Form", u"~/config.xml", None))
        self.lineEdit_volume_seqfile.setPlaceholderText(QCoreApplication.translate("Form", u"~/sequence.mha", None))
        self.pushButton_volume_volfileBrowse.setText("")
        self.pushButton_volume_seqfileBrowse.setText("")
        self.pushButton_volume_outputdirBrowse.setText("")
        self.label_7.setText(QCoreApplication.translate("Form", u"Threshold", None))
        self.pushButton_volume_reconstruct.setText(QCoreApplication.translate("Form", u"Reconstruct Volume", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"Sequence File", None))
        self.pushButton_volume_configfileBrowse.setText("")
        self.label_4.setText(QCoreApplication.translate("Form", u"Load Volume", None))
        self.pushButton_volume_volload.setText(QCoreApplication.translate("Form", u"Load", None))
    # retranslateUi

