import React, { useState } from 'react';
import PropTypes from 'prop-types';
import Transition from '../utils/Transition';

function Dropdown({ options, defaultOption }) {
  const [selectedOption, setSelectedOption] = useState(defaultOption);
  const [isOpen, setIsOpen] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const handleOptionClick = (option) => {
    setSelectedOption(option);
    setIsOpen(false);
  };

  const toggleDropdown = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className="relative">
      <button
        type="button"
        className="bg-gray-400 hover:bg-slate-700	 px-4 py-2 flex items-center transition duration-150 ease-in-out"
        onClick={toggleDropdown}
      >
        {selectedOption}
        <svg className="w-3 h-3 fill-current text-gray-500 cursor-pointer ml-1 shrink-0" viewBox="0 0 12 12" xmlns="http://www.w3.org/2000/svg">
          <path d="M10.28 4.305L5.989 8.598 1.695 4.305A1 1 0 00.28 5.72l5 5a1 1 0 001.414 0l5-5a1 1 0 10-1.414-1.414z" />
        </svg>
      </button>
      {isOpen && (
        <ul className="absolute top-full left-0 z-10 bg-gray-800 py-2 mt-1 rounded-sm">
          {options.map((option) => (
            <li key={option}>
              <button
                type="button"
                className="block px-4 py-2 text-gray-300 hover:bg-gray-700 focus:bg-gray-700 focus:outline-none"
                onClick={() => handleOptionClick(option)}
              >
                {option}
              </button>
            </li>
          ))}
        </ul>
      )}

      <Transition
        show={dropdownOpen}
        enter="transition ease-out duration-100 transform"
        enterFrom="opacity-0 scale-95"
        enterTo="opacity-100 scale-100"
        leave="transition ease-in duration-75 transform"
        leaveFrom="opacity-100 scale-100"
        leaveTo="opacity-0 scale-95"
      >
      <div className="origin-top-right absolute right-0 mt-2 w-64 rounded-md shadow-lg bg-gray-900 ring-1 ring-black ring-opacity-5 divide-y divide-gray-300 focus:outline-none">
          {options.map((option) => (
            <div
              key={option}
              className="py-1"
              onClick={() => handleOptionClick(option)}
            >
              <span className="block px-4 py-2 text-sm text-white hover:bg-gray-700 cursor-pointer">
                {option}
              </span>
            </div>
          ))}
        </div>
      </Transition>
    </div>

  );
}

Dropdown.propTypes = {
  options: PropTypes.arrayOf(PropTypes.string).isRequired,
  defaultOption: PropTypes.string.isRequired,
};

export default Dropdown;
