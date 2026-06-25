# 银发守护者系统 - 视觉样式配置

class Colors:
    """色彩体系规范 - 温暖科技风格"""
    
    # 主色调
    PRIMARY_BLUE = '#4A90E2'      # 守护蓝 - logo、主按钮、标题栏
    PRIMARY_BLUE_RGB = (74, 144, 226)
    
    # 辅助色
    ACCENT_ORANGE = '#FF9F43'     # 温暖橙 - 告警提示、强调信息
    ACCENT_ORANGE_RGB = (255, 159, 67)
    
    # 中性色
    NEUTRAL_LIGHT = '#F5F7FA'     # 浅灰 - 背景色、卡片底色
    NEUTRAL_LIGHT_RGB = (245, 247, 250)
    
    NEUTRAL_DARK = '#333333'      # 深灰 - 正文文字
    NEUTRAL_DARK_RGB = (51, 51, 51)
    
    NEUTRAL_MEDIUM = '#889096'    # 中灰 - 辅助文字、边框线
    NEUTRAL_MEDIUM_RGB = (136, 144, 150)
    
    NEUTRAL_WHITE = '#FFFFFF'     # 白色 - 按钮文字、图标背景
    NEUTRAL_WHITE_RGB = (255, 255, 255)
    
    # 告警色
    ALERT_RED = '#E53E3E'         # 紧急红 - 跌倒、生命体征异常等一级告警
    ALERT_RED_RGB = (229, 62, 62)
    
    ALERT_YELLOW = '#ECC94B'      # 预警黄 - 异常行为、情绪低落等二级告警
    ALERT_YELLOW_RGB = (236, 201, 75)
    
    # 状态色
    STATUS_GREEN = '#38A169'      # 成功绿 - 正常状态、已处理告警
    STATUS_GREEN_RGB = (56, 161, 105)
    
    STATUS_INFO = '#3182CE'       # 信息蓝 - 信息提示、普通通知
    STATUS_INFO_RGB = (49, 130, 206)
    
    @classmethod
    def get_color_by_alarm_level(cls, level):
        """根据告警级别获取颜色"""
        if level == 'critical' or level == 'fall':
            return cls.ALERT_RED
        elif level == 'warning' or level == 'abnormal':
            return cls.ALERT_YELLOW
        elif level == 'info':
            return cls.STATUS_INFO
        else:
            return cls.NEUTRAL_MEDIUM

class Fonts:
    """字体规范"""
    
    FAMILY_HEADING = 'Microsoft YaHei, SimHei, sans-serif'    # 标题字体
    FAMILY_BODY = 'Microsoft YaHei, SimSun, sans-serif'       # 正文字体
    FAMILY_MONO = 'Consolas, Monaco, monospace'              # 数字/等宽字体
    
    # 字号规范 (px)
    SIZE_H1 = 32       # 主标题
    SIZE_H2 = 24       # 副标题
    SIZE_H3 = 18       # 小标题
    SIZE_BODY = 14     # 正文
    SIZE_CAPTION = 12  # 辅助文字
    SIZE_DATA = 28     # 数据展示
    
    # 字重
    WEIGHT_BOLD = 700
    WEIGHT_SEMIBOLD = 600
    WEIGHT_MEDIUM = 500
    WEIGHT_REGULAR = 400
    WEIGHT_LIGHT = 300
    
    # 行高
    LINE_HEIGHT_HEADING = 1.2
    LINE_HEIGHT_BODY = 1.5
    LINE_HEIGHT_DATA = 1.3

class Sizes:
    """尺寸规范"""
    
    # 圆角
    ROUNDING_SMALL = 6      # 按钮、小元素
    ROUNDING_MEDIUM = 12    # 卡片、弹窗
    ROUNDING_LARGE = 16     # 面板、大容器
    
    # 间距
    SPACING_XS = 4
    SPACING_SM = 8
    SPACING_MD = 16
    SPACING_LG = 24
    SPACING_XL = 32
    
    # 卡片尺寸
    CARD_ALARM_WIDTH = 320
    CARD_ALARM_HEIGHT = 120
    
    CARD_DASHBOARD_WIDTH = 200
    CARD_DASHBOARD_HEIGHT = 120
    
    # 图标尺寸
    ICON_SMALL = 24
    ICON_MEDIUM = 32
    ICON_LARGE = 48
    
    # 阴影（Qt不支持box-shadow，留空）
    SHADOW_CARD = ''
    SHADOW_BUTTON = ''

class StyleSheet:
    """Qt样式表生成器"""
    
    @classmethod
    def get_main_window_style(cls):
        return f"""
            QMainWindow {{
                background-color: {Colors.NEUTRAL_LIGHT};
            }}
        """
    
    @classmethod
    def get_video_widget_style(cls):
        return f"""
            QWidget {{
                background-color: #1a1a1a;
                border-radius: {Sizes.ROUNDING_MEDIUM}px;
                border: 2px solid {Colors.NEUTRAL_MEDIUM};
            }}
        """
    
    @classmethod
    def get_alarm_list_style(cls):
        return f"""
            QListWidget {{
                background-color: {Colors.NEUTRAL_WHITE};
                border: 1px solid {Colors.NEUTRAL_MEDIUM};
                border-radius: {Sizes.ROUNDING_MEDIUM}px;
                padding: {Sizes.SPACING_SM}px;
            }}
            QListWidget::item {{
                padding: {Sizes.SPACING_MD}px;
                border-bottom: 1px solid {Colors.NEUTRAL_LIGHT};
                border-radius: {Sizes.ROUNDING_SMALL}px;
                margin-bottom: {Sizes.SPACING_XS}px;
            }}
            QListWidget::item:selected {{
                background-color: {Colors.PRIMARY_BLUE};
                color: {Colors.NEUTRAL_WHITE};
            }}
        """
    
    @classmethod
    def get_alarm_item_style(cls, level='normal'):
        color = Colors.get_color_by_alarm_level(level)
        return f"""
            background-color: rgba({color[1:]}, 0.1);
            border-left: 4px solid {color};
        """
    
    @classmethod
    def get_button_style(cls, type='primary'):
        if type == 'primary':
            bg_color = Colors.PRIMARY_BLUE
            text_color = Colors.NEUTRAL_WHITE
            hover_bg = '#3a80d2'
        elif type == 'success':
            bg_color = Colors.STATUS_GREEN
            text_color = Colors.NEUTRAL_WHITE
            hover_bg = '#2a9159'
        elif type == 'danger':
            bg_color = Colors.ALERT_RED
            text_color = Colors.NEUTRAL_WHITE
            hover_bg = '#d52e2e'
        else:
            bg_color = Colors.NEUTRAL_MEDIUM
            text_color = Colors.NEUTRAL_WHITE
            hover_bg = '#788086'
        
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: {text_color};
                border: none;
                border-radius: {Sizes.ROUNDING_SMALL}px;
                padding: {Sizes.SPACING_SM}px {Sizes.SPACING_MD}px;
                font-family: {Fonts.FAMILY_BODY};
                font-size: {Fonts.SIZE_BODY}px;
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
            }}
            QPushButton:pressed {{
                background-color: {bg_color};
            }}
            QPushButton:disabled {{
                background-color: {Colors.NEUTRAL_LIGHT};
                color: {Colors.NEUTRAL_MEDIUM};
            }}
        """
    
    @classmethod
    def get_progress_bar_style(cls, color_type='normal'):
        if color_type == 'success':
            color = Colors.STATUS_GREEN
        elif color_type == 'warning':
            color = Colors.ALERT_YELLOW
        elif color_type == 'danger':
            color = Colors.ALERT_RED
        else:
            color = Colors.PRIMARY_BLUE
        
        return f"""
            QProgressBar {{
                border: none;
                border-radius: {Sizes.ROUNDING_SMALL}px;
                background-color: {Colors.NEUTRAL_LIGHT};
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: {Sizes.ROUNDING_SMALL}px;
            }}
        """
    
    @classmethod
    def get_label_style(cls, type='body'):
        if type == 'heading':
            font_size = Fonts.SIZE_H3
            font_weight = Fonts.WEIGHT_BOLD
            color = Colors.NEUTRAL_DARK
        elif type == 'data':
            font_size = Fonts.SIZE_DATA
            font_weight = Fonts.WEIGHT_BOLD
            color = Colors.PRIMARY_BLUE
        elif type == 'caption':
            font_size = Fonts.SIZE_CAPTION
            font_weight = Fonts.WEIGHT_REGULAR
            color = Colors.NEUTRAL_MEDIUM
        else:
            font_size = Fonts.SIZE_BODY
            font_weight = Fonts.WEIGHT_REGULAR
            color = Colors.NEUTRAL_DARK
        
        return f"""
            QLabel {{
                font-family: {Fonts.FAMILY_BODY};
                font-size: {font_size}px;
                font-weight: {font_weight};
                color: {color};
            }}
        """
    
    @classmethod
    def get_status_bar_style(cls):
        return f"""
            QWidget {{
                background-color: {Colors.NEUTRAL_WHITE};
                border-top: 1px solid {Colors.NEUTRAL_LIGHT};
                padding: {Sizes.SPACING_SM}px {Sizes.SPACING_MD}px;
            }}
            QLabel {{
                font-family: {Fonts.FAMILY_MONO};
                font-size: {Fonts.SIZE_CAPTION}px;
                color: {Colors.NEUTRAL_MEDIUM};
            }}
        """
    
    @classmethod
    def get_camera_list_style(cls):
        return f"""
            QListWidget {{
                background-color: {Colors.NEUTRAL_WHITE};
                border: 1px solid {Colors.NEUTRAL_MEDIUM};
                border-radius: {Sizes.ROUNDING_MEDIUM}px;
                padding: {Sizes.SPACING_SM}px;
            }}
            QListWidget::item {{
                padding: {Sizes.SPACING_MD}px;
                border-radius: {Sizes.ROUNDING_SMALL}px;
                margin-bottom: {Sizes.SPACING_XS}px;
            }}
            QListWidget::item:hover {{
                background-color: rgba({Colors.PRIMARY_BLUE[1:]}, 0.1);
            }}
            QListWidget::item:selected {{
                background-color: rgba({Colors.PRIMARY_BLUE[1:]}, 0.2);
                border: 2px solid {Colors.PRIMARY_BLUE};
            }}
        """