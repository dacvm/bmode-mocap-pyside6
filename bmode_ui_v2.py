# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'bmode_v2.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(600, 400)
        self.horizontalLayout_3 = QHBoxLayout(Form)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_bmode = QHBoxLayout()
        self.horizontalLayout_bmode.setObjectName(u"horizontalLayout_bmode")
        self.verticalLayout_menu = QVBoxLayout()
        self.verticalLayout_menu.setObjectName(u"verticalLayout_menu")
        self.comboBox_bmode_streamOption = QComboBox(Form)
        self.comboBox_bmode_streamOption.addItem("")
        self.comboBox_bmode_streamOption.addItem("")
        self.comboBox_bmode_streamOption.addItem("")
        self.comboBox_bmode_streamOption.setObjectName(u"comboBox_bmode_streamOption")

        self.verticalLayout_menu.addWidget(self.comboBox_bmode_streamOption)

        self.comboBox_bmode_streamPort = QComboBox(Form)
        self.comboBox_bmode_streamPort.setObjectName(u"comboBox_bmode_streamPort")
        self.comboBox_bmode_streamPort.setEnabled(True)

        self.verticalLayout_menu.addWidget(self.comboBox_bmode_streamPort)

        self.horizontalLayout_calib = QHBoxLayout()
        self.horizontalLayout_calib.setObjectName(u"horizontalLayout_calib")
        self.lineEdit_bmode_calibPath = QLineEdit(Form)
        self.lineEdit_bmode_calibPath.setObjectName(u"lineEdit_bmode_calibPath")
        self.lineEdit_bmode_calibPath.setEnabled(False)

        self.horizontalLayout_calib.addWidget(self.lineEdit_bmode_calibPath)

        self.pushButton_bmode_calibClear = QPushButton(Form)
        self.pushButton_bmode_calibClear.setObjectName(u"pushButton_bmode_calibClear")
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.EditClear))
        self.pushButton_bmode_calibClear.setIcon(icon)

        self.horizontalLayout_calib.addWidget(self.pushButton_bmode_calibClear)

        self.pushButton_bmode_calibBrowse = QPushButton(Form)
        self.pushButton_bmode_calibBrowse.setObjectName(u"pushButton_bmode_calibBrowse")
        self.pushButton_bmode_calibBrowse.setEnabled(True)
        icon1 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.DocumentOpen))
        self.pushButton_bmode_calibBrowse.setIcon(icon1)

        self.horizontalLayout_calib.addWidget(self.pushButton_bmode_calibBrowse)


        self.verticalLayout_menu.addLayout(self.horizontalLayout_calib)

        self.pushButton_bmode_openStream = QPushButton(Form)
        self.pushButton_bmode_openStream.setObjectName(u"pushButton_bmode_openStream")
        self.pushButton_bmode_openStream.setEnabled(True)

        self.verticalLayout_menu.addWidget(self.pushButton_bmode_openStream)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_menu.addItem(self.verticalSpacer)


        self.horizontalLayout_bmode.addLayout(self.verticalLayout_menu)

        self.label_bmode_image = QLabel(Form)
        self.label_bmode_image.setObjectName(u"label_bmode_image")

        self.horizontalLayout_bmode.addWidget(self.label_bmode_image)

        self.horizontalLayout_bmode.setStretch(0, 1)
        self.horizontalLayout_bmode.setStretch(1, 3)

        self.horizontalLayout_3.addLayout(self.horizontalLayout_bmode)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.comboBox_bmode_streamOption.setItemText(0, QCoreApplication.translate("Form", u"Stream Image", None))
        self.comboBox_bmode_streamOption.setItemText(1, QCoreApplication.translate("Form", u"Stream Screen (Other)", None))
        self.comboBox_bmode_streamOption.setItemText(2, QCoreApplication.translate("Form", u"Stream Screen (This PC)", None))

        self.comboBox_bmode_streamPort.setPlaceholderText(QCoreApplication.translate("Form", u"Select USB Port", None))
        self.lineEdit_bmode_calibPath.setPlaceholderText(QCoreApplication.translate("Form", u"D:/", None))
        self.pushButton_bmode_calibClear.setText("")
        self.pushButton_bmode_calibBrowse.setText(QCoreApplication.translate("Form", u"Calib File", None))
        self.pushButton_bmode_openStream.setText(QCoreApplication.translate("Form", u"Open Stream", None))
        self.label_bmode_image.setText("")
    # retranslateUi

