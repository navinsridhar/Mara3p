/**
 ==============================================================================
 Copyright 2019, Jonathan Zrake

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 ==============================================================================
*/




#pragma once
#include <optional>
#include "core_numeric_array.hpp"
#include "core_bsp_tree.hpp"




//=============================================================================
namespace bsp { // bqo := binary tree, quad-tree, oct-tree




/**
 * @brief      Return a boolean sequence {a} representing a number:
 *
 *             value = a[0] * 2^0 + a[1] * 2^1 + ...
 *
 * @param[in]  value     The number to represent
 *
 * @tparam     BitCount  The number of bits to include
 *
 * @return     A boolean sequence, with the most significant bit at the
 *             beginning.
 */
template<std::size_t BitCount>
auto binary_repr(std::size_t value)
{
    return map(numeric::range<BitCount>(), [value] (auto n) { return bool(value & (1 << n)); });
}




/**
 * @brief      Turn a binary representation of a number into a 64-bit unsigned
 *             integer.
 *
 * @param[in]  bits      The bits (as returned from binary_repr)
 *
 * @tparam     BitCount  The number of bits
 *
 * @return     The decimal representation
 */
template<std::size_t BitCount>
unsigned long to_integral(numeric::array_t<bool, BitCount> bits)
{
    return sum(apply_to(map(numeric::range<BitCount>(), [] (auto e) { return [e] (bool y) { return (1 << e) * y; }; }), bits));
}




/**
 * @brief      A struct that identifies a node's global position in the tree:
 *             its level, and its coordinates with respect to the origin at its
 *             level.
 *
 * @tparam     Rank  The rank of the tree to be indexed (having ratio 2^Rank)
 */
template<bsp::uint Rank>
struct tree_index_t
{
    unsigned long level = 0;
    numeric::array_t<unsigned long, Rank> coordinates = {};
};




/**
 * @brief      Determine if this is a valid index (whether it is in-bounds on
 *             its level).
 *
 * @param[in]  i     The index
 *
 * @tparam     R     The rank
 *
 * @return     True or false
 */
template<std::size_t R>
bool valid(const tree_index_t<R>& i)
{
    return all(map(i.coordinates, [level=i.level] (auto j) { return j < unsigned(1 << level); }));
}




/**
 * @brief      Return true if this index is at level 0 (in which case an
 *             exception is thrown if the coordinates were not all zero).
 *
 * @param[in]  i     The index
 *
 * @tparam     R     The rank
 *
 * @return     True or false
 */
template<std::size_t R>
bool check_root(const tree_index_t<R>& i)
{
    if (i.level == 0)
    {
        if (any(i.coordinates))
        {
            throw std::invalid_argument("bsp::tree_index_t (invalid tree index)");
        }
        return true;
    }
    return false;
}




/**
 * @brief      Transform this index, so that it points to the same node as it
 *             does now, but as seen by the child of this node which is in the
 *             direction of the target node.
 *
 * @param[in]  i     The index
 *
 * @tparam     R     The rank
 *
 * @return     The index with level - 1 and the coordinates offset according to
 *             the orthant value
 */
template<std::size_t R>
tree_index_t<R> advance_level(const tree_index_t<R>& i)
{
    return {i.level - 1, i.coordinates - orthant(i) * (1 << (i.level - 1))};
}




/**
 * @brief      Return this index as it is relative to the parent block.
 *
 * @param[in]  i     The index
 *
 * @tparam     R     The rank
 *
 * @return     The index as seen by the parent.
 */
template<std::size_t R>
tree_index_t<R> relative_to_parent(const tree_index_t<R>& i)
{
    return {1, map(i.coordinates, [] (auto c) { return c % 2; })};
}




/**
 * @brief      Return the tree index at level - 1 which contains this one as a
 *             child.
 *
 * @param[in]  i     The index
 *
 * @tparam     R     The rank
 *
 * @return     The parent index
 */
template<std::size_t R>
tree_index_t<R> parent_index(const tree_index_t<R>& i)
{
    return {i.level - 1, i.coordinates / 2};
}




/**
 * @brief      Return a sequence of tree indexes corresponding to the 2^R
 *             indexes of this node's children.
 *
 * @param[in]  i     The index
 *
 * @tparam     R     The rank
 *
 * @return     A sequence of tree indexes
 */
template<std::size_t R>
numeric::array_t<tree_index_t<R>, 1 << R> child_indexes(const tree_index_t<R>& i)
{
    return map(numeric::range<1 << R>(), [&i] (auto j) -> tree_index_t<R>
    {
        return {i.level + 1, i.coordinates * 2 + binary_repr<R>(j)};
    });
}




/**
 * @brief      Helper function to give the number of children per node at the
 *             given Rank
 *
 * @param[in]  <unnamed>  { parameter_description }
 *
 * @tparam     R          { description }
 *
 * @return     An integer
 */
template<std::size_t R>
std::size_t child_count(const tree_index_t<R>&)
{
    return 1 << R;
}




/**
 * @brief      Return the linear index (between 0 and 2^Rank, non-inclusive) of
 *             this index in its parent.
 *
 * @param[in]  i     { parameter_description }
 *
 * @tparam     R     { description }
 *
 * @return     The sibling index
 */
template<std::size_t R>
std::size_t sibling_index(const tree_index_t<R>& i)
{
    return to_integral(orthant(relative_to_parent(i)));
}




/**
 * @brief      Determines whether the specified i is last child.
 *
 * @param[in]  i     { parameter_description }
 *
 * @tparam     R     { description }
 *
 * @return     True if the specified i is last child, False otherwise.
 */
template<std::size_t R>
bool is_last_child(const tree_index_t<R>& i)
{
    return i.level == 0 || sibling_index(i) == child_count(i) - 1;
}




/**
 * @brief      Return the next sibling's index. Throws std::out_of_range if
 *             next sibling does not exist
 *
 * @return     An index
 */
template<std::size_t R>
tree_index_t<R> next_sibling(const tree_index_t<R>& i)
{
    if (is_last_child(i))
    {
        throw std::out_of_range("bsp::next_sibling (is the last child)");
    }
    return tree_index_t<R>{i.level, parent_index(i).coordinates * 2 + binary_repr<R>(sibling_index(i) + 1)};
}




/**
 * @brief      Return a new index with the same coordinates but a different
 *             level.
 *
 * @param[in]  i          The index
 * @param[in]  new_level  The level of the returned index
 *
 * @tparam     R          The rank
 *
 * @return     A new index
 */
template<std::size_t R>
tree_index_t<R> with_level(const tree_index_t<R>& i, unsigned long new_level)
{
    return {new_level, i.coordinates};
}




/**
 * @brief      Return a new index with the same coordinates but a different
 *             level.
 *
 * @param[in]  i                The index
 * @param[in]  new_coordinates  The coordinates of the returned index
 *
 * @tparam     R                The rank
 *
 * @return     A new index
 */
template<std::size_t R>
tree_index_t<R> with_coordinates(const tree_index_t<R>& i, numeric::array_t<unsigned long, R> new_coordinates)
{
    return {i.level, new_coordinates};
}




/**
 * @brief      Return the orthant (ray, quadrant, octant) of this index.
 *
 * @param[in]  i     The index
 *
 * @tparam     R     The rank
 *
 * @return     A sequence of bool's
 *
 * @note       https://en.wikipedia.org/wiki/Orthant
 */
template<std::size_t R>
numeric::array_t<bool, R> orthant(const tree_index_t<R> &i)
{
    return map(i.coordinates, [level=i.level] (auto x) -> bool { return x / (1 << (level - 1)); });
}




/**
 * @brief      Return an index shifted up on the given axis. The result index is
 *             wrapped between 0 and the maximum value at this level.
 *
 * @param[in]  i     The index
 * @param[in]  a     The axis to shift on
 *
 * @tparam     R     The rank
 *
 * @return     A new index, on the same level
 */
template<std::size_t R> tree_index_t<R> next_on(const tree_index_t<R>& i, std::size_t a) { return {i.level, update(i.coordinates, a, [L=i.level] (auto j) { return (j + (1 << L) + 1) % (1 << L); })}; }
template<std::size_t R> tree_index_t<R> prev_on(const tree_index_t<R>& i, std::size_t a) { return {i.level, update(i.coordinates, a, [L=i.level] (auto j) { return (j + (1 << L) - 1) % (1 << L); })}; }




//=============================================================================
template<std::size_t R> bool operator==(const tree_index_t<R>& a, const tree_index_t<R>& b) { return a.level == b.level && a.coordinates == b.coordinates; }
template<std::size_t R> bool operator!=(const tree_index_t<R>& a, const tree_index_t<R>& b) { return a.level != b.level || a.coordinates != b.coordinates; }
template<std::size_t R> bool operator< (const tree_index_t<R>& a, const tree_index_t<R>& b) { return a.level != b.level ? a.level < b.level : a.coordinates.impl < b.coordinates.impl; }
template<std::size_t R> bool operator> (const tree_index_t<R>& a, const tree_index_t<R>& b) { return a.level != b.level ? a.level > b.level : a.coordinates.impl > b.coordinates.impl; }




/**
 * @brief      Return a default-constructed tree_index_t
 *
 * @tparam     Rank  The rank of the tree
 *
 * @return     A new tree index
 */
template<std::size_t Rank>
tree_index_t<Rank> tree_index()
{
    return tree_index_t<Rank>{};
}




/**
 * @brief      Return the child of the given (non-leaf) tree node. This function
 *             wraps the bsp method based on linear indexes.
 *
 * @param[in]  tree          The tree (throws out_of_range if not a leaf)
 * @param[in]  orthant       The orthant (ray, quadrant, octant)
 *
 * @tparam     ValueType     The value type of the tree
 * @tparam     ChildrenType  The tree's children provider type
 * @tparam     Rank          The rank of the tree
 *
 * @return     A tree node that is a child of the given node
 */
template<typename ValueType, typename ChildrenType, bsp::uint Rank>
auto child_at(const bsp::tree_t<ValueType, ChildrenType, 1 << Rank>& tree, numeric::array_t<bool, Rank> orthant)
{
    return child_at(tree, to_integral(orthant));
}




/**
 * @brief      Return a node at an arbitrary depth below the given node. If no
 *             node exists there, throws out_of_range.
 *
 * @param[in]  tree          The tree to index
 * @param[in]  index         The index into the tree
 *
 * @tparam     ValueType     The value type of the tree
 * @tparam     ChildrenType  The tree's children provider type
 * @tparam     Rank          The rank of the tree
 *
 * @return     A tree node that is a descendant of the given node
 */
template<typename ValueType, typename ChildrenType, bsp::uint Rank>
auto node_at(const bsp::tree_t<ValueType, ChildrenType, 1 << Rank>& tree, tree_index_t<Rank> index)
-> bsp::tree_t<ValueType, ChildrenType, 1 << Rank>
{
    return check_root(index) ? tree : node_at(child_at(tree, orthant(index)), advance_level(index));
}




/**
 * @brief      Determine whether a node exists at an arbitrary depth below the
 *             given node.
 *
 * @param[in]  tree          The tree to check
 * @param[in]  index         The index into the tree
 *
 * @tparam     ValueType     The value type of the tree
 * @tparam     ChildrenType  The tree's children provider type
 * @tparam     Rank          The rank of the tree
 *
 * @return     True or false
 */
template<typename ValueType, typename ChildrenType, bsp::uint Rank>
bool contains(const bsp::tree_t<ValueType, ChildrenType, 1 << Rank>& tree, tree_index_t<Rank> index)
{
    return has_value(tree) ? check_root(index) : contains(child_at(tree, orthant(index)), advance_level(index));
}




template<bsp::uint Ratio> struct rank_for_ratio_t {};
template<> struct rank_for_ratio_t<2> { static const bsp::uint value = 1; };
template<> struct rank_for_ratio_t<4> { static const bsp::uint value = 2; };
template<> struct rank_for_ratio_t<8> { static const bsp::uint value = 3; };




/**
 * @brief      Functions implementing the sequence protocol. These make it
 *             possible to adapt a bqo tree to a sequence via seq::adapt.
 *
 * @param[in]  tree          The tree to adapt to a sequence
 * @param[in]  current       The starting index (used internally)
 *
 * @tparam     ValueType     The value type of the tree
 * @tparam     ChildrenType  The tree's children provider type
 *
 * @return     An optional to the tree start, as expected by sequence operators
 */
template<typename ValueType, typename ChildrenType, bsp::uint Ratio, bsp::uint Rank=rank_for_ratio_t<Ratio>::value>
std::optional<tree_index_t<Rank>> start(bsp::tree_t<ValueType, ChildrenType, Ratio> tree, tree_index_t<Rank> current={})
{
    return has_value(tree) ? current : start(child_at(tree, 0), child_indexes(current).at(0));
}

template<typename ValueType, typename ChildrenType, bsp::uint Ratio, bsp::uint Rank=rank_for_ratio_t<Ratio>::value>
std::optional<tree_index_t<Rank>> next(bsp::tree_t<ValueType, ChildrenType, Ratio> tree, tree_index_t<Rank> current)
{
    if (check_root(current))
    {
        return {};
    }

    if (is_last_child(current))
    {
        auto ancestor = parent_index(current);

        while (is_last_child(ancestor))
        {
            if (check_root(ancestor))
            {
                return {};
            }
            ancestor = parent_index(ancestor);
        }
        current = next_sibling(ancestor);

        while (! contains(tree, current))
        {
            current = child_indexes(current).at(0);
        }
        return current;
    }
    current = next_sibling(current);

    while (! contains(tree, current))
    {
        current = child_indexes(current).at(0);
    }
    return current;
}

template<typename ValueType, typename ChildrenType, bsp::uint Ratio, bsp::uint Rank=rank_for_ratio_t<Ratio>::value>
auto obtain(bsp::tree_t<ValueType, ChildrenType, Ratio> tree, tree_index_t<Rank> index)
{
    return value(node_at(tree, index));
}




//=============================================================================
inline auto uniform_quadtree(bsp::uint depth)
{
    auto branch_function = [] (auto i) { return child_indexes(i); };
    auto result = bsp::just<4>(bsp::tree_index<2>());

    while (depth--)
    {
        result = branch_through(result, branch_function);
    }
    return result;
}

inline auto uniform_octree(bsp::uint depth)
{
    auto branch_function = [] (auto i) { return child_indexes(i); };
    auto result = bsp::just<8>(bsp::tree_index<3>());

    while (depth--)
    {
        result = branch_through(result, branch_function);
    }
    return result;
}

} // namespace bsp





//=============================================================================
#ifdef DO_UNIT_TESTS
#include "core_unit_test.hpp"
#include "core_sequence.hpp"




//=============================================================================
inline void test_bqo_tree()
{
    require(bsp::to_integral(numeric::array(false, false, false)) == 0);
    require(bsp::to_integral(numeric::array(true, true, true)) == 7);
    require(bsp::to_integral(numeric::array(true, false, false)) == 1); // the 2^0 bit is on the left
    require(bsp::to_integral(bsp::binary_repr<3>(6)) == 6);

    require(orthant(bsp::tree_index_t<3>{1, {0, 0, 0}}) == numeric::array(false, false, false));
    require(orthant(bsp::tree_index_t<3>{1, {0, 0, 1}}) == numeric::array(false, false, true));
    require(orthant(bsp::tree_index_t<3>{1, {0, 1, 0}}) == numeric::array(false, true, false));
    require(orthant(bsp::tree_index_t<3>{1, {1, 0, 0}}) == numeric::array(true, false, false));
    require(orthant(bsp::tree_index_t<3>{2, {0, 0, 1}}) == numeric::array(false, false, false));
    require(orthant(bsp::tree_index_t<3>{2, {0, 1, 0}}) == numeric::array(false, false, false));
    require(orthant(bsp::tree_index_t<3>{2, {1, 0, 0}}) == numeric::array(false, false, false));
    require(orthant(bsp::tree_index_t<3>{2, {0, 0, 2}}) == numeric::array(false, false, true));
    require(orthant(bsp::tree_index_t<3>{2, {0, 2, 0}}) == numeric::array(false, true, false));
    require(orthant(bsp::tree_index_t<3>{2, {2, 0, 0}}) == numeric::array(true, false, false));

    require_throws(next_on(bsp::tree_index<3>(), 3));
    require(  next_on(with_level(bsp::tree_index<3>(), 3), 0).coordinates == numeric::array(1, 0, 0));
    require(  next_on(with_level(bsp::tree_index<3>(), 3), 1).coordinates == numeric::array(0, 1, 0));
    require(  next_on(with_level(bsp::tree_index<3>(), 3), 2).coordinates == numeric::array(0, 0, 1));
    require(  prev_on(with_level(bsp::tree_index<3>(), 3), 0).coordinates == numeric::array(7, 0, 0));
    require(  prev_on(with_level(bsp::tree_index<3>(), 3), 1).coordinates == numeric::array(0, 7, 0));
    require(  prev_on(with_level(bsp::tree_index<3>(), 3), 2).coordinates == numeric::array(0, 0, 7));
    require(  valid(with_coordinates(with_level(bsp::tree_index<3>(), 3), {0, 0, 7})));
    require(! valid(with_coordinates(with_level(bsp::tree_index<3>(), 3), {0, 0, 8})));

    auto branch_function = [] (auto i) { return child_indexes(i); };
    auto tree64 = branch_through(branch_through(bsp::just<8>(bsp::tree_index<3>()), branch_function), branch_function);
    auto index = with_coordinates(with_level(bsp::tree_index<3>(), 2), {0, 1, 2});

    require(size(bsp::from(child_indexes(bsp::tree_index<3>()))) == 8);
    require(size(branch(bsp::just<8>(bsp::tree_index<3>()), branch_function)) == 8);
    require(size(tree64) == 64);
    require(has_value(bsp::node_at(tree64, index)));
    require(  contains(tree64, index));
    require(! contains(tree64, child_indexes(index)[0]));
    require_throws(bsp::node_at(tree64, child_indexes(index)[0]));
}

#endif // DO_UNIT_TESTS
