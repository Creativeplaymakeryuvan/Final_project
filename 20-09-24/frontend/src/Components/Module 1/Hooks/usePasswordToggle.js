import react, { useState } from 'react';
import { FaEye } from "react-icons/fa";
import { FaEyeSlash } from "react-icons/fa";
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import { FaE } from 'react-icons/fa6';

const usePasswordToggle = () => {
    const [visible, setvisible]=useState(false)
    const Icon = (
        <FontAwesomeIcon icon={ visible ? {FaEyeSlash} : {FaEye}} / >
    )
    const InputType = visible ? "text" : "password";
    return[
        InputType,Icon
    ]
}

export default usePasswordToggle