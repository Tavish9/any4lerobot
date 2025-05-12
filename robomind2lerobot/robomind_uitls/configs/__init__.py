from .agilex_3rgb import AgileX_3RGB_Config
from .franka_1rgb import Franka_1RGB_Config
from .franka_3rgb import Franka_3RGB_Config
from .franka_fr3_dual_arm import Franka_Fr3_Dual_Arm_Config
from .tienkung_gello_1rgb import Tien_Kung_Gello_1RGB_Config
from .tienkung_prod1_gello_1rgb import Tien_Kung_Prod1_Gello_1RGB_Config
from .tienkung_xsens_1rgb import Tien_Kung_Xsens_1RGB_Config
from .ur_1rgb import UR_1RGB_Config

ROBOMIND_CONFIG = dict(
    agilex_3rgb=AgileX_3RGB_Config,
    franka_1rgb=Franka_1RGB_Config,
    franka_3rgb=Franka_3RGB_Config,
    franka_fr3_dual=Franka_Fr3_Dual_Arm_Config,
    sim_franka_3rgb="",
    sim_tienkung_1rgb="",
    tienkung_gello_1rgb=Tien_Kung_Gello_1RGB_Config,
    tienkung_prod1_gello_1rgb=Tien_Kung_Prod1_Gello_1RGB_Config,
    tienkung_xsens_1rgb=Tien_Kung_Xsens_1RGB_Config,
    ur_1rgb=UR_1RGB_Config,
)
